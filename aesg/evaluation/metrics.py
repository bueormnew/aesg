"""Domain-specific evaluation metrics for AESG models.

Provides automatic domain detection and metric computation for text,
image, and classification architectures. All metrics are implemented
in pure PyTorch/numpy with no external library dependencies.
"""

import warnings
import math
from typing import Dict, List, Union
from collections import Counter

import torch
import torch.nn as nn


class Evaluator:
    """Automatic domain-specific evaluation metrics.

    Detects model architecture type and computes the appropriate set
    of metrics without requiring manual configuration.

    Supported domains:
        - text: BLEU, ROUGE, Accuracy, Perplexity
        - image: PSNR, SSIM, MSE, MAE
        - classification: Accuracy, Precision, Recall, F1
    """

    @staticmethod
    def detect_domain(model: nn.Module) -> str:
        """Detect architecture domain based on model class name.

        Args:
            model: A PyTorch nn.Module instance.

        Returns:
            One of "text", "image", "classification", or "unknown".
        """
        class_name = type(model).__name__

        # Text architectures
        text_patterns = ["Transformer", "GRUText", "LSTMText", "Seq2Seq", "DecoderLM"]
        for pattern in text_patterns:
            if pattern in class_name:
                return "text"

        # Classification architectures
        if "Classifier" in class_name:
            return "classification"

        # Image architectures
        image_patterns = ["Colorization", "Autoencoder"]
        for pattern in image_patterns:
            if pattern in class_name:
                return "image"

        return "unknown"

    @staticmethod
    def compute_text_metrics(
        predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """Compute text generation metrics.

        Args:
            predictions: List of predicted text strings.
            references: List of reference text strings.

        Returns:
            Dictionary with BLEU, ROUGE, Accuracy, and Perplexity scores.
        """
        if not predictions or not references:
            warnings.warn(
                "Empty predictions or references provided. "
                "Cannot compute text metrics."
            )
            return {}

        n = min(len(predictions), len(references))
        predictions = predictions[:n]
        references = references[:n]

        # BLEU: simplified 4-gram precision
        bleu = Evaluator._compute_bleu(predictions, references)

        # ROUGE: simplified ROUGE-1 (unigram recall)
        rouge = Evaluator._compute_rouge(predictions, references)

        # Accuracy: exact match ratio
        exact_matches = sum(
            1 for pred, ref in zip(predictions, references) if pred == ref
        )
        accuracy = exact_matches / n

        # Perplexity: approximate via cross-entropy over character-level
        perplexity = Evaluator._compute_perplexity(predictions, references)

        return {
            "bleu": float(bleu),
            "rouge": float(rouge),
            "accuracy": float(accuracy),
            "perplexity": float(perplexity),
        }

    @staticmethod
    def compute_image_metrics(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute image quality metrics.

        Assumes pixel values are in [0, 1] range.

        Args:
            predictions: Predicted images tensor (B, C, H, W).
            targets: Ground truth images tensor (B, C, H, W).

        Returns:
            Dictionary with PSNR, SSIM, MSE, and MAE scores.
        """
        if predictions.numel() == 0 or targets.numel() == 0:
            warnings.warn(
                "Empty predictions or targets provided. "
                "Cannot compute image metrics."
            )
            return {}

        predictions = predictions.float()
        targets = targets.float()

        # MSE: mean squared error
        mse = torch.mean((predictions - targets) ** 2).item()

        # MAE: mean absolute error
        mae = torch.mean(torch.abs(predictions - targets)).item()

        # PSNR: 10 * log10(1.0 / MSE), assumes [0, 1] range
        if mse > 0:
            psnr = 10.0 * math.log10(1.0 / mse)
        else:
            psnr = float("inf")

        # SSIM: simplified structural similarity using mean, variance, covariance
        ssim = Evaluator._compute_ssim(predictions, targets)

        return {
            "psnr": float(psnr),
            "ssim": float(ssim),
            "mse": float(mse),
            "mae": float(mae),
        }

    @staticmethod
    def compute_classification_metrics(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """Compute classification metrics with macro averaging.

        Args:
            predictions: Logits tensor of shape (B, C) where C is num classes.
            targets: Class index tensor of shape (B,).

        Returns:
            Dictionary with Accuracy, Precision, Recall, and F1 scores.
        """
        if predictions.numel() == 0 or targets.numel() == 0:
            warnings.warn(
                "Empty predictions or targets provided. "
                "Cannot compute classification metrics."
            )
            return {}

        # Get predicted class indices
        pred_classes = predictions.argmax(dim=1)
        num_classes = predictions.shape[1]

        # Accuracy: correct / total
        correct = (pred_classes == targets).sum().item()
        total = targets.shape[0]
        accuracy = correct / total if total > 0 else 0.0

        # Per-class precision and recall for macro average
        precisions = []
        recalls = []

        for c in range(num_classes):
            pred_positive = (pred_classes == c)
            actual_positive = (targets == c)

            true_positive = (pred_positive & actual_positive).sum().item()
            pred_positive_count = pred_positive.sum().item()
            actual_positive_count = actual_positive.sum().item()

            # Precision for class c
            if pred_positive_count > 0:
                precisions.append(true_positive / pred_positive_count)
            else:
                precisions.append(0.0)

            # Recall for class c
            if actual_positive_count > 0:
                recalls.append(true_positive / actual_positive_count)
            else:
                recalls.append(0.0)

        # Macro average
        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0

        # F1: harmonic mean of precision and recall
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def evaluate(
        self,
        model: nn.Module,
        predictions: Union[List[str], torch.Tensor],
        references: Union[List[str], torch.Tensor],
    ) -> Dict[str, float]:
        """Evaluate model using domain-appropriate metrics.

        Automatically detects the model domain and applies the
        corresponding metric set.

        Args:
            model: The model being evaluated (used for domain detection).
            predictions: Model outputs (strings for text, tensors otherwise).
            references: Ground truth values.

        Returns:
            Dictionary of metric name → float value.
            Empty dict if domain is unsupported or inputs are empty.
        """
        # Check for empty inputs
        if isinstance(predictions, list) and len(predictions) == 0:
            warnings.warn(
                "Empty input data provided. Cannot compute metrics."
            )
            return {}
        if isinstance(predictions, torch.Tensor) and predictions.numel() == 0:
            warnings.warn(
                "Empty input data provided. Cannot compute metrics."
            )
            return {}

        domain = self.detect_domain(model)

        if domain == "text":
            return self.compute_text_metrics(predictions, references)
        elif domain == "image":
            return self.compute_image_metrics(predictions, references)
        elif domain == "classification":
            return self.compute_classification_metrics(predictions, references)
        else:
            warnings.warn(
                f"Unsupported architecture '{type(model).__name__}' for "
                f"automatic metrics. Supported domains: text, image, "
                f"classification."
            )
            return {}

    # ----------------------------------------------------------------
    # Private helper methods
    # ----------------------------------------------------------------

    @staticmethod
    def _compute_bleu(predictions: List[str], references: List[str]) -> float:
        """Simplified corpus-level 4-gram BLEU score.

        Computes modified n-gram precision for n=1..4 and applies
        brevity penalty.
        """
        max_n = 4
        clip_counts = [0] * max_n
        total_counts = [0] * max_n
        pred_length = 0
        ref_length = 0

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            pred_length += len(pred_tokens)
            ref_length += len(ref_tokens)

            for n in range(1, max_n + 1):
                pred_ngrams = Evaluator._get_ngrams(pred_tokens, n)
                ref_ngrams = Evaluator._get_ngrams(ref_tokens, n)

                # Clipped counts
                for ngram, count in pred_ngrams.items():
                    clip_counts[n - 1] += min(count, ref_ngrams.get(ngram, 0))
                total_counts[n - 1] += sum(pred_ngrams.values())

        # Modified precision for each n
        precisions = []
        for n in range(max_n):
            if total_counts[n] > 0:
                precisions.append(clip_counts[n] / total_counts[n])
            else:
                precisions.append(0.0)

        # If any precision is zero, BLEU is zero
        if any(p == 0.0 for p in precisions):
            return 0.0

        # Geometric mean of precisions with uniform weights
        log_avg = sum(math.log(p) for p in precisions) / max_n
        geo_mean = math.exp(log_avg)

        # Brevity penalty
        if pred_length <= 0:
            return 0.0
        if pred_length < ref_length:
            bp = math.exp(1.0 - ref_length / pred_length)
        else:
            bp = 1.0

        return bp * geo_mean

    @staticmethod
    def _compute_rouge(predictions: List[str], references: List[str]) -> float:
        """Simplified ROUGE-1 (unigram recall) score.

        Computes average unigram recall across all prediction-reference pairs.
        """
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            if not ref_tokens:
                scores.append(0.0)
                continue

            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)

            # Count overlapping unigrams (clipped)
            overlap = 0
            for token, count in ref_counter.items():
                overlap += min(count, pred_counter.get(token, 0))

            recall = overlap / sum(ref_counter.values())
            scores.append(recall)

        return sum(scores) / len(scores) if scores else 0.0

    @staticmethod
    def _compute_perplexity(
        predictions: List[str], references: List[str]
    ) -> float:
        """Approximate perplexity using character-level cross-entropy.

        Uses a uniform character distribution as baseline, computing
        how well predictions match references character-by-character.
        """
        total_log_prob = 0.0
        total_chars = 0

        for pred, ref in zip(predictions, references):
            if not ref:
                continue

            # Build character frequency from reference
            ref_counter = Counter(ref)
            vocab_size = len(ref_counter)

            if vocab_size == 0:
                continue

            # For each character in prediction, compute log probability
            # based on reference character distribution
            for ch in pred:
                freq = ref_counter.get(ch, 0)
                # Smoothed probability
                prob = (freq + 1) / (len(ref) + vocab_size)
                total_log_prob += math.log(prob)
                total_chars += 1

        if total_chars == 0:
            return float("inf")

        # Cross-entropy (negative average log probability)
        cross_entropy = -total_log_prob / total_chars

        # Perplexity = exp(cross_entropy)
        return math.exp(cross_entropy)

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Dict[str, int]:
        """Extract n-grams from token list and return counts."""
        ngrams: Dict[str, int] = {}
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    @staticmethod
    def _compute_ssim(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Simplified SSIM using mean, variance, covariance approach.

        Computes per-image SSIM and returns the batch average.
        Constants C1 and C2 stabilize division with weak denominators.
        """
        # Constants for stability (based on dynamic range of 1.0)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Flatten spatial dimensions for per-image computation
        # predictions/targets: (B, C, H, W)
        batch_size = predictions.shape[0]
        ssim_values = []

        for i in range(batch_size):
            x = predictions[i].flatten().float()
            y = targets[i].flatten().float()

            mu_x = x.mean()
            mu_y = y.mean()

            sigma_x_sq = ((x - mu_x) ** 2).mean()
            sigma_y_sq = ((y - mu_y) ** 2).mean()
            sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

            numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
            denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)

            ssim_val = (numerator / denominator).item()
            ssim_values.append(ssim_val)

        return sum(ssim_values) / len(ssim_values) if ssim_values else 0.0
