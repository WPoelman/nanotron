import math
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from nanotron.config import GenerationArgs, LanguageMetricsArgs, TokenizerArgs
from nanotron.generation.decode import GenerationInput, decode_text
from nanotron.parallel import ParallelContext


class LanguageMetricsLogger:
    """Extended metrics logger for per-language evaluation"""

    def __init__(self, config: LanguageMetricsArgs, tokenizer_config: TokenizerArgs, parallel_context: ParallelContext):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.tokenizer_name_or_path)
        self.parallel_context = parallel_context

        # go from "<corpus>/<language>"" -> sequences
        if self.config.test_datasets:
            self.test_datasets: Dict[str, List[str]] = {}
            self._load_test_datasets()

    def _load_test_datasets(self):
        """Load test datasets for each language"""
        for path in self.config.test_datasets:
            path = Path(path)  # TODO(Wessel) figure out how the yaml parser doesn't break on a list of Path objects...
            with open(path, "r", encoding="utf-8") as f:
                lang = path.stem
                corpus = path.parent.name
                self.test_datasets[f"{corpus}/{lang}"] = [line.strip() for line in f.readlines()]

    def compute_test_set_metrics(self, model: torch.nn.Module, current_step: int) -> Dict[str, torch.Tensor]:
        """Compute NLL, PPL, BPC, MRR for each language"""
        metrics = {"current_step": current_step}
        model.eval()

        with torch.no_grad():
            for key in self.config.languages.keys():
                metrics.update(self._compute_language_test_metrics(model, key))

        if self.config.results_path:
            self._write_metrics_to_csv(metrics)

        model.train()
        return metrics

    def _compute_language_test_metrics(self, model: torch.nn.Module, key: str) -> Dict[str, torch.Tensor]:
        """Compute all test-set metrics for a specific language"""
        test_data = self.test_datasets[key]
        batch_size = self.config.batch_size

        total_nll = 0.0
        total_tokens = 0
        total_chars = 0
        mrr_scores = []

        for i in range(0, len(test_data), batch_size):
            batch_texts = test_data[i : i + batch_size]
            batch_metrics = self._compute_batch_metrics(model, batch_texts)

            total_nll += batch_metrics["nll"]
            total_tokens += batch_metrics["tokens"]
            total_chars += batch_metrics["chars"]
            mrr_scores.extend(batch_metrics["mrr_scores"])

        # Compute final metrics (after batches)
        avg_nll = total_nll / len(test_data)
        ppl = math.exp(avg_nll)
        bpc = (total_nll * math.log(2)) / total_chars  # Convert to bits per character
        mrr = np.mean(mrr_scores) if mrr_scores else 0.0

        return {
            f"{key}/nll": torch.tensor(avg_nll),
            f"{key}/ppl": torch.tensor(ppl),
            f"{key}/bpc": torch.tensor(bpc),
            f"{key}/mrr": torch.tensor(mrr),
        }

    def _compute_batch_metrics(self, model: torch.nn.Module, texts: List[str]) -> Dict:
        """Compute metrics for a batch of texts"""
        # Tokenize batch
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute NLL
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention = attention_mask[..., 1:].contiguous()

        # Compute loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        losses = losses.view(shift_labels.shape)

        # Mask out padding tokens
        masked_losses = losses * shift_attention.float()

        # Compute MRR scores
        mrr_scores = self._compute_mrr_scores(shift_logits, shift_labels, shift_attention)

        # Aggregate metrics
        total_nll = masked_losses.sum().item()
        total_tokens = shift_attention.sum().item()
        total_chars = sum(len(text) for text in texts)

        return {"nll": total_nll, "tokens": total_tokens, "chars": total_chars, "mrr_scores": mrr_scores}

    def _compute_mrr_scores(
        self, logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[float]:
        """Compute Mean Reciprocal Rank scores"""
        mrr_scores = []

        # Get probabilities
        probs = F.softmax(logits, dim=-1)

        for i in range(logits.size(0)):  # For each sequence in batch
            for j in range(logits.size(1)):  # For each position
                if attention_mask[i, j] == 0:
                    continue

                true_token = labels[i, j].item()
                token_probs = probs[i, j]

                # Get rank of true token
                _, sorted_indices = torch.sort(token_probs, descending=True)
                rank = (sorted_indices == true_token).nonzero(as_tuple=True)[0].item() + 1

                mrr_scores.append(1.0 / rank)

        return mrr_scores

    def compute_generative_metrics(self, model: torch.nn.Module, current_step: int) -> Dict[str, torch.Tensor]:
        """Compute Zipf's law and Heaps law metrics"""
        metrics = {"current_step": current_step}
        model.eval()

        with torch.no_grad():
            for lang in self.config.languages:
                # Generate tokens for this language
                generated_tokens = self._generate_tokens_for_language(model, lang)

                # TODO(Wessel): probably don't use correlation, maybe R² or something else.
                zipf_score = self._compute_zipf_adherence(generated_tokens)
                metrics[f"{lang}/zipf_adherence"] = torch.tensor(zipf_score)

                heaps_score = self._compute_heaps_adherence(generated_tokens)
                metrics[f"{lang}/heaps_adherence"] = torch.tensor(heaps_score)

        if self.config.results_path:
            self._write_metrics_to_csv(metrics)

        model.train()
        return metrics

    def _generate_tokens_for_language(self, model: torch.nn.Module) -> List[int]:
        """Generate tokens by sampling directly from the model without prompt"""

        generation_config = GenerationArgs(sampler="greedy", temperature=0.8, use_cache=False)

        # Start with just BOS token or empty input
        if hasattr(self.tokenizer, "bos_token") and self.tokenizer.bos_token is not None:
            prompt = self.tokenizer.bos_token
        else:
            prompt = ""

        # TODO(Wessel): make sure we actually sample enough tokens from the model
        outputs = decode_text(
            input_iter=[GenerationInput(text=prompt)],
            tokenizer=self.tokenizer,
            model=model,
            parallel_context=self.parallel_context,
            max_new_tokens=self.language_metrics_config.generation_samples,
            max_micro_batch_size=1,
            generation_config=generation_config,
            tokenizer_config=None,
        )

        for output in outputs:
            if isinstance(output.generation_ids, torch.Tensor):
                return output.generation_ids.cpu().tolist()

        return []

    def _compute_zipf_adherence(self, tokens: List[int]) -> float:
        """Compute how well the token distribution follows Zipf's law"""
        # Count token frequencies
        token_counts = Counter(tokens)
        frequencies = sorted(token_counts.values(), reverse=True)

        if len(frequencies) < 2:
            return 0.0

        # Compute Zipf's law fit (frequency ∝ 1/rank)
        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)

        # Linear regression to find slope (should be close to -1 for perfect Zipf)
        correlation = np.corrcoef(log_ranks, log_freqs)[0, 1]
        return abs(correlation)  # Higher correlation = better Zipf adherence

    def _compute_heaps_adherence(self, tokens: List[int]) -> float:
        """Compute how well vocabulary growth follows Heaps' law"""
        unique_tokens = set()
        vocab_sizes = []

        # Track vocabulary growth
        for i, token in enumerate(tokens, 1):
            unique_tokens.add(token)
            if i % 100 == 0:  # Sample every 100 tokens
                vocab_sizes.append(len(unique_tokens))

        if len(vocab_sizes) < 2:
            return 0.0

        # Heaps' law: V(n) = K * n^β where β is typically 0.4-0.6
        n_values = np.arange(100, len(tokens) + 1, 100)[: len(vocab_sizes)]
        log_n = np.log(n_values)
        log_v = np.log(vocab_sizes)

        # Linear regression to find β
        correlation = np.corrcoef(log_n, log_v)[0, 1]
        return abs(correlation)

    def _write_metrics_to_csv(self, metrics: Dict[str, torch.Tensor]):
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        pd.DataFrame(metrics).to_csv(
            self.config.results_path, mode="a", index=False, header=not self.config.results_path.exists()
        )
