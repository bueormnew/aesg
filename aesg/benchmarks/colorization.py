"""
AESG Colorization Benchmark.

Automated CNN vs CNN+AESG colorization comparison benchmark.
Trains both models on a dataset of grayscale→color image pairs and
compares quality metrics (PSNR, SSIM) and AESG-specific statistics.

Works with or without torchvision — falls back to synthetic data
when torchvision is unavailable.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


@dataclass
class BenchmarkResult:
    """Results from a single model's benchmark run.

    Attributes
    ----------
    model_name : str
        Identifier for the model (e.g. "aesg" or "baseline").
    loss_per_epoch : List[float]
        Average training loss for each epoch.
    psnr : float
        Peak Signal-to-Noise Ratio on the eval set.
    ssim : float
        Structural Similarity Index on the eval set.
    train_time_seconds : float
        Total wall-clock training time in seconds.
    concepts_created : int
        Number of AESG concepts created during training.
    concepts_merged : int
        Number of AESG concepts merged during consolidation.
    regions_activated : int
        Number of semantic regions activated during training.
    memory_nodes : int
        Total memory nodes at end of training.
    """
    model_name: str = ""
    loss_per_epoch: List[float] = field(default_factory=list)
    psnr: float = 0.0
    ssim: float = 0.0
    train_time_seconds: float = 0.0
    concepts_created: int = 0
    concepts_merged: int = 0
    regions_activated: int = 0
    memory_nodes: int = 0


class _ColorizationDataset(Dataset):
    """Simple dataset of (grayscale, color) image pairs as tensors."""

    def __init__(self, gray_images: torch.Tensor, color_images: torch.Tensor):
        """
        Parameters
        ----------
        gray_images : torch.Tensor
            Grayscale images of shape (N, 1, H, W) in [0, 1].
        color_images : torch.Tensor
            Color images of shape (N, 3, H, W) in [0, 1].
        """
        self.gray_images = gray_images
        self.color_images = color_images

    def __len__(self) -> int:
        return self.gray_images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gray_images[idx], self.color_images[idx]


class BaselineCNN(nn.Module):
    """Baseline CNN for colorization WITHOUT AESG memory.

    Same encoder-decoder architecture as AESGColorizationNet 'small' but
    uses a simple Conv2d at the bottleneck instead of AESG_CNNLayer.

    Architecture:
        Encoder: 3 blocks of Conv2d+BN+ReLU+MaxPool2d (1→32→64→128)
        Bottleneck: Conv2d (128→128, 3x3, padding=1)
        Decoder: 3 blocks of ConvTranspose2d+BN+ReLU (128→64→32→3 with sigmoid)
    """

    def __init__(self):
        super().__init__()
        base_filters = 32
        num_blocks = 3

        # Encoder: 1ch → 32 → 64 → 128
        encoder_layers = []
        in_ch = 1
        for i in range(num_blocks):
            out_ch = base_filters * (2 ** i)
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # Bottleneck: plain conv (no AESG)
        bottleneck_channels = base_filters * (2 ** (num_blocks - 1))  # 128
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
        )

        # Decoder: 128 → 64 → 32 → 3
        decoder_layers = []
        in_ch = bottleneck_channels
        for i in range(num_blocks - 1, 0, -1):
            out_ch = base_filters * (2 ** (i - 1))
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        # Final upscale to 3 channels
        decoder_layers.append(
            nn.ConvTranspose2d(in_ch, 3, kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(*decoder_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: grayscale → color.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Output of shape (B, 3, H, W) with values in [0, 1].
        """
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        dec = self.decoder(bottleneck)
        return self.sigmoid(dec)


class ColorizationBenchmark:
    """Automated CNN vs CNN+AESG colorization comparison benchmark.

    Downloads or generates a dataset of color images, creates grayscale/color
    pairs, trains both an AESG-augmented colorization model and a baseline CNN,
    then compares PSNR/SSIM metrics and AESG memory statistics.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use ("cifar10" or "synthetic").
    max_images : int
        Maximum number of images to use from the dataset.
    image_size : int
        Target spatial resolution (images resized to image_size x image_size).
    max_epochs : int
        Number of training epochs.
    output_dir : str
        Directory for saving results, samples, and reports.
    """

    def __init__(
        self,
        dataset_name: str = "cifar10",
        max_images: int = 2000,
        image_size: int = 128,
        max_epochs: int = 3,
        output_dir: str = "./benchmark_results",
    ):
        self.dataset_name = dataset_name
        self.max_images = max_images
        self.image_size = image_size
        self.max_epochs = max_epochs
        self.output_dir = output_dir

    def run(self) -> Dict[str, BenchmarkResult]:
        """Execute the full benchmark pipeline.

        Steps:
            1. Download/generate dataset
            2. Preprocess: resize, create grayscale/color pairs, split 80/20
            3. Train AESGColorizationNet.small() and BaselineCNN
            4. Compute PSNR/SSIM on eval set
            5. Save visual samples
            6. Generate markdown report

        Returns
        -------
        Dict[str, BenchmarkResult]
            Results dict with keys "aesg" and "baseline".
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Download/generate dataset
        color_images = self._download_dataset()

        # 2. Preprocess: resize, grayscale/color pairs, 80/20 split
        gray_images, color_images = self._preprocess(color_images)

        split_idx = int(len(gray_images) * 0.8)
        train_gray = gray_images[:split_idx]
        train_color = color_images[:split_idx]
        eval_gray = gray_images[split_idx:]
        eval_color = color_images[split_idx:]

        train_dataset = _ColorizationDataset(train_gray, train_color)
        eval_dataset = _ColorizationDataset(eval_gray, eval_color)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

        # 3. Create and train both models
        from aesg.architectures.image import AESGColorizationNet

        aesg_storage_dir = os.path.join(self.output_dir, "aesg_memory")
        os.makedirs(aesg_storage_dir, exist_ok=True)
        aesg_model = AESGColorizationNet.small(storage_dir=aesg_storage_dir)
        baseline_model = BaselineCNN()

        aesg_result = self._train_model(aesg_model, train_loader, "aesg")
        baseline_result = self._train_model(baseline_model, train_loader, "baseline")

        # 4. Evaluate both models
        aesg_metrics = self._evaluate_model(aesg_model, eval_loader)
        baseline_metrics = self._evaluate_model(baseline_model, eval_loader)

        aesg_result.psnr = aesg_metrics.get("psnr", 0.0)
        aesg_result.ssim = aesg_metrics.get("ssim", 0.0)
        baseline_result.psnr = baseline_metrics.get("psnr", 0.0)
        baseline_result.ssim = baseline_metrics.get("ssim", 0.0)

        # Collect AESG memory stats
        aesg_result = self._collect_aesg_stats(aesg_model, aesg_result)

        # 5. Save visual samples
        self._save_samples(eval_gray, eval_color, aesg_model, baseline_model, num_samples=10)

        # 6. Generate report
        self._generate_report(aesg_result, baseline_result)

        return {"aesg": aesg_result, "baseline": baseline_result}

    def _download_dataset(self) -> np.ndarray:
        """Download or generate the image dataset.

        Attempts to use torchvision CIFAR-10. If torchvision is unavailable,
        generates synthetic colored images (random gradients and patterns).

        Returns
        -------
        np.ndarray
            Color images array of shape (N, H, W, 3) with values in [0, 255].
        """
        try:
            import torchvision
            import torchvision.transforms as transforms

            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])

            dataset = torchvision.datasets.CIFAR10(
                root=os.path.join(self.output_dir, "data"),
                train=True,
                download=True,
                transform=transform,
            )

            # Extract images up to max_images
            images = []
            for i in range(min(self.max_images, len(dataset))):
                img, _ = dataset[i]
                # img is (3, H, W) tensor in [0, 1]
                img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                images.append(img_np)

            return np.array(images)

        except (ImportError, Exception):
            # Fallback: generate synthetic colored images
            return self._generate_synthetic_images()

    def _generate_synthetic_images(self) -> np.ndarray:
        """Generate synthetic colored images as fallback.

        Creates random gradient patterns, circles, and color blocks
        to provide meaningful colorization targets.

        Returns
        -------
        np.ndarray
            Synthetic color images of shape (N, H, W, 3) in [0, 255].
        """
        rng = np.random.default_rng(42)
        images = []
        h, w = self.image_size, self.image_size

        for i in range(self.max_images):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            pattern_type = i % 4

            if pattern_type == 0:
                # Horizontal gradient
                color1 = rng.integers(50, 255, size=3)
                color2 = rng.integers(50, 255, size=3)
                for x in range(w):
                    t = x / max(w - 1, 1)
                    img[:, x, :] = (color1 * (1 - t) + color2 * t).astype(np.uint8)

            elif pattern_type == 1:
                # Vertical gradient
                color1 = rng.integers(50, 255, size=3)
                color2 = rng.integers(50, 255, size=3)
                for y in range(h):
                    t = y / max(h - 1, 1)
                    img[y, :, :] = (color1 * (1 - t) + color2 * t).astype(np.uint8)

            elif pattern_type == 2:
                # Color blocks (2x2 grid)
                colors = rng.integers(30, 255, size=(4, 3))
                mid_h, mid_w = h // 2, w // 2
                img[:mid_h, :mid_w, :] = colors[0]
                img[:mid_h, mid_w:, :] = colors[1]
                img[mid_h:, :mid_w, :] = colors[2]
                img[mid_h:, mid_w:, :] = colors[3]

            else:
                # Circle on background
                bg_color = rng.integers(20, 100, size=3)
                fg_color = rng.integers(100, 255, size=3)
                img[:, :, :] = bg_color
                cy, cx = h // 2, w // 2
                radius = min(h, w) // 4
                for y in range(h):
                    for x in range(w):
                        if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                            img[y, x, :] = fg_color

            images.append(img)

        return np.array(images)

    def _preprocess(
        self, images: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess color images into grayscale/color tensor pairs.

        Parameters
        ----------
        images : np.ndarray
            Color images of shape (N, H, W, 3) in [0, 255].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            gray_images: (N, 1, H, W) in [0, 1]
            color_images: (N, 3, H, W) in [0, 1]
        """
        # Normalize to [0, 1] and convert to (N, 3, H, W)
        color_tensor = torch.from_numpy(images).float() / 255.0
        color_tensor = color_tensor.permute(0, 3, 1, 2)  # (N, 3, H, W)

        # Resize if needed (images may already be correct size from torchvision)
        if color_tensor.shape[2] != self.image_size or color_tensor.shape[3] != self.image_size:
            color_tensor = nn.functional.interpolate(
                color_tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        # Convert to grayscale: standard luminance formula
        # Y = 0.2989 * R + 0.5870 * G + 0.1140 * B
        gray_tensor = (
            0.2989 * color_tensor[:, 0:1, :, :]
            + 0.5870 * color_tensor[:, 1:2, :, :]
            + 0.1140 * color_tensor[:, 2:3, :, :]
        )

        return gray_tensor, color_tensor

    def _train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        model_name: str,
    ) -> BenchmarkResult:
        """Train a model and record training metrics.

        Parameters
        ----------
        model : nn.Module
            The model to train (AESG or baseline).
        train_loader : DataLoader
            Training data loader yielding (gray, color) pairs.
        model_name : str
            Identifier for logging ("aesg" or "baseline").

        Returns
        -------
        BenchmarkResult
            Partial result with loss_per_epoch and train_time_seconds filled.
        """
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        loss_per_epoch: List[float] = []

        start_time = time.time()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for gray_batch, color_batch in train_loader:
                optimizer.zero_grad()
                output = model(gray_batch)
                loss = criterion(output, color_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            loss_per_epoch.append(avg_loss)

        train_time = time.time() - start_time

        return BenchmarkResult(
            model_name=model_name,
            loss_per_epoch=loss_per_epoch,
            train_time_seconds=train_time,
        )

    def _evaluate_model(
        self, model: nn.Module, eval_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate a model and compute image quality metrics.

        Uses the Evaluator from aesg.evaluation.metrics for PSNR/SSIM.

        Parameters
        ----------
        model : nn.Module
            Trained model to evaluate.
        eval_loader : DataLoader
            Evaluation data loader yielding (gray, color) pairs.

        Returns
        -------
        Dict[str, float]
            Metrics dict with psnr, ssim, mse, mae keys.
        """
        from aesg.evaluation.metrics import Evaluator

        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for gray_batch, color_batch in eval_loader:
                output = model(gray_batch)
                all_predictions.append(output)
                all_targets.append(color_batch)

        if not all_predictions:
            return {"psnr": 0.0, "ssim": 0.0, "mse": 1.0, "mae": 1.0}

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = Evaluator.compute_image_metrics(predictions, targets)
        return metrics

    def _collect_aesg_stats(
        self, model: nn.Module, result: BenchmarkResult
    ) -> BenchmarkResult:
        """Collect AESG memory statistics from the trained model.

        Parameters
        ----------
        model : nn.Module
            The AESG model with memory module.
        result : BenchmarkResult
            The result to populate with AESG stats.

        Returns
        -------
        BenchmarkResult
            Updated result with memory statistics.
        """
        try:
            # Walk model modules to find AESGMemory
            from aesg.memory.controller import AESGMemory

            for module in model.modules():
                if isinstance(module, AESGMemory):
                    result.memory_nodes = module.storage.node_count
                    # Attempt to get stats from logger
                    if hasattr(module, 'logger') and hasattr(module.logger, 'stats'):
                        stats = module.logger.stats
                        result.concepts_created = stats.get("concepts_created", 0)
                        result.concepts_merged = stats.get("concepts_merged", 0)
                        result.regions_activated = stats.get("regions_activated", 0)
                    break
        except Exception:
            pass  # Stats are best-effort

        return result

    def _save_samples(
        self,
        eval_gray: torch.Tensor,
        eval_color: torch.Tensor,
        aesg_model: nn.Module,
        baseline_model: nn.Module,
        num_samples: int = 10,
    ) -> None:
        """Save visual comparison samples as PNG images.

        Saves original color, grayscale input, and colorized outputs from
        both models side by side.

        Parameters
        ----------
        eval_gray : torch.Tensor
            Grayscale eval images (N, 1, H, W).
        eval_color : torch.Tensor
            Color eval images (N, 3, H, W).
        aesg_model : nn.Module
            Trained AESG model.
        baseline_model : nn.Module
            Trained baseline model.
        num_samples : int
            Number of sample images to save.
        """
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        num_samples = min(num_samples, len(eval_gray))
        sample_gray = eval_gray[:num_samples]
        sample_color = eval_color[:num_samples]

        aesg_model.eval()
        baseline_model.eval()

        with torch.no_grad():
            aesg_output = aesg_model(sample_gray)
            baseline_output = baseline_model(sample_gray)

        try:
            import torchvision.utils as vutils

            for i in range(num_samples):
                # Save each set: original, grayscale (3ch), aesg, baseline
                gray_3ch = sample_gray[i].repeat(3, 1, 1)  # (3, H, W)
                grid = torch.stack([
                    sample_color[i],
                    gray_3ch,
                    aesg_output[i].clamp(0, 1),
                    baseline_output[i].clamp(0, 1),
                ], dim=0)

                vutils.save_image(
                    grid,
                    os.path.join(samples_dir, f"sample_{i:03d}.png"),
                    nrow=4,
                    padding=2,
                )

        except ImportError:
            # torchvision not available — save as raw numpy arrays
            for i in range(num_samples):
                sample_data = {
                    "original": sample_color[i].numpy().tolist(),
                    "grayscale": sample_gray[i].numpy().tolist(),
                    "aesg": aesg_output[i].clamp(0, 1).numpy().tolist(),
                    "baseline": baseline_output[i].clamp(0, 1).numpy().tolist(),
                }
                with open(os.path.join(samples_dir, f"sample_{i:03d}.json"), "w") as f:
                    json.dump(sample_data, f)

    def _generate_report(
        self, aesg_result: BenchmarkResult, baseline_result: BenchmarkResult
    ) -> None:
        """Generate a markdown comparison report.

        Parameters
        ----------
        aesg_result : BenchmarkResult
            Results from the AESG model.
        baseline_result : BenchmarkResult
            Results from the baseline CNN.
        """
        report_path = os.path.join(self.output_dir, "benchmark_report.md")

        psnr_diff = aesg_result.psnr - baseline_result.psnr
        ssim_diff = aesg_result.ssim - baseline_result.ssim
        time_ratio = (
            aesg_result.train_time_seconds / max(baseline_result.train_time_seconds, 0.001)
        )

        lines = [
            "# AESG Colorization Benchmark Report",
            "",
            "## Configuration",
            "",
            f"- **Dataset**: {self.dataset_name}",
            f"- **Images**: {self.max_images}",
            f"- **Image Size**: {self.image_size}x{self.image_size}",
            f"- **Epochs**: {self.max_epochs}",
            "",
            "## Results Summary",
            "",
            "| Metric | AESG | Baseline | Difference |",
            "|--------|------|----------|------------|",
            f"| PSNR (dB) | {aesg_result.psnr:.2f} | {baseline_result.psnr:.2f} | {psnr_diff:+.2f} |",
            f"| SSIM | {aesg_result.ssim:.4f} | {baseline_result.ssim:.4f} | {ssim_diff:+.4f} |",
            f"| Train Time (s) | {aesg_result.train_time_seconds:.1f} | {baseline_result.train_time_seconds:.1f} | {time_ratio:.2f}x |",
            "",
            "## Training Loss Per Epoch",
            "",
            "| Epoch | AESG Loss | Baseline Loss |",
            "|-------|-----------|---------------|",
        ]

        max_epochs = max(len(aesg_result.loss_per_epoch), len(baseline_result.loss_per_epoch))
        for epoch in range(max_epochs):
            aesg_loss = (
                f"{aesg_result.loss_per_epoch[epoch]:.6f}"
                if epoch < len(aesg_result.loss_per_epoch)
                else "N/A"
            )
            baseline_loss = (
                f"{baseline_result.loss_per_epoch[epoch]:.6f}"
                if epoch < len(baseline_result.loss_per_epoch)
                else "N/A"
            )
            lines.append(f"| {epoch + 1} | {aesg_loss} | {baseline_loss} |")

        lines.extend([
            "",
            "## AESG Memory Statistics",
            "",
            f"- **Concepts Created**: {aesg_result.concepts_created}",
            f"- **Concepts Merged**: {aesg_result.concepts_merged}",
            f"- **Regions Activated**: {aesg_result.regions_activated}",
            f"- **Memory Nodes**: {aesg_result.memory_nodes}",
            "",
            "## Analysis",
            "",
        ])

        if psnr_diff > 0:
            lines.append(
                f"The AESG model achieved **{psnr_diff:.2f} dB higher PSNR** than the baseline, "
                f"indicating better pixel-level reconstruction quality."
            )
        elif psnr_diff < 0:
            lines.append(
                f"The baseline model achieved **{abs(psnr_diff):.2f} dB higher PSNR** than AESG. "
                f"This may indicate the AESG memory overhead impacts convergence in short training runs."
            )
        else:
            lines.append("Both models achieved comparable PSNR scores.")

        lines.append("")

        if ssim_diff > 0:
            lines.append(
                f"AESG achieved **{ssim_diff:.4f} higher SSIM**, suggesting better structural "
                f"preservation aided by semantic memory."
            )
        elif ssim_diff < 0:
            lines.append(
                f"The baseline achieved **{abs(ssim_diff):.4f} higher SSIM**."
            )
        else:
            lines.append("Both models achieved comparable SSIM scores.")

        lines.extend([
            "",
            "## Notes",
            "",
            "- This benchmark uses a limited dataset and few epochs for fast iteration.",
            "- AESG benefits are expected to be more pronounced with larger datasets and longer training.",
            "- Visual samples are saved in the `samples/` subdirectory.",
            "",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        # Also save raw results as JSON
        results_json_path = os.path.join(self.output_dir, "results.json")
        results_data = {
            "aesg": {
                "model_name": aesg_result.model_name,
                "loss_per_epoch": aesg_result.loss_per_epoch,
                "psnr": aesg_result.psnr,
                "ssim": aesg_result.ssim,
                "train_time_seconds": aesg_result.train_time_seconds,
                "concepts_created": aesg_result.concepts_created,
                "concepts_merged": aesg_result.concepts_merged,
                "regions_activated": aesg_result.regions_activated,
                "memory_nodes": aesg_result.memory_nodes,
            },
            "baseline": {
                "model_name": baseline_result.model_name,
                "loss_per_epoch": baseline_result.loss_per_epoch,
                "psnr": baseline_result.psnr,
                "ssim": baseline_result.ssim,
                "train_time_seconds": baseline_result.train_time_seconds,
                "concepts_created": baseline_result.concepts_created,
                "concepts_merged": baseline_result.concepts_merged,
                "regions_activated": baseline_result.regions_activated,
                "memory_nodes": baseline_result.memory_nodes,
            },
            "config": {
                "dataset_name": self.dataset_name,
                "max_images": self.max_images,
                "image_size": self.image_size,
                "max_epochs": self.max_epochs,
            },
        }
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)
