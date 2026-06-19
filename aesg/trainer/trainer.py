"""
AESG Professional Trainer.

HuggingFace-style trainer managing the dual training loop (backpropagation
+ topology update) with lifecycle callbacks, multi-format data adaptation,
and full checkpoint persistence including AESG memory state.

This module provides the AESGTrainer class which supersedes the legacy
DualTrainer for production use.
"""

import json
import os
import shutil
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from aesg.exceptions import AESGStorageError, AESGTrainingError


class _ListDataset(Dataset):
    """Wraps a list of (input, target) tuples as a PyTorch Dataset."""

    def __init__(self, data: List):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int):
        item = self._data[idx]
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return item[0], item[1]
        raise AESGTrainingError(
            "List items must be (input, target) tuples of length 2."
        )


class AESGTrainer:
    """Professional trainer for AESG-augmented models.

    Manages the dual training loop (neural backpropagation + graph topology
    update), callback lifecycle, data format adaptation, and full model
    persistence including AESG memory state.

    Parameters
    ----------
    model : nn.Module
        The model to train. May contain an AESGMemory submodule.
    optimizer : Optional[torch.optim.Optimizer]
        Optimizer for neural weight updates. Defaults to Adam.
    criterion : Optional[nn.Module]
        Loss function. Defaults to CrossEntropyLoss.
    callbacks : Optional[List]
        List of Callback instances for lifecycle hooks.
    device : str
        Device to run training on ("cpu" or "cuda").
    batch_size : int
        Default batch size for data adaptation.
    learning_rate : float
        Learning rate for auto-created optimizer.

    Examples
    --------
    >>> model = AESGGRUText.small(vocab_size=1000)
    >>> trainer = AESGTrainer(model)
    >>> losses = trainer.fit(train_data, epochs=5)
    >>> metrics = trainer.evaluate(test_data)
    >>> trainer.save("./checkpoints/epoch_5")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        callbacks: Optional[List] = None,
        device: str = "cpu",
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Auto-create optimizer if not provided
        self.optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=learning_rate
        )

        # Auto-create criterion if not provided
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Callbacks (default empty)
        self.callbacks = callbacks or []

        # Find AESGMemory module in model
        self._memory_module = self._find_memory(model)
        if self._memory_module is None:
            warnings.warn(
                "No AESGMemory module found in model. "
                "Operating in neural-only mode (no topology updates).",
                stacklevel=2,
            )

        # Training state
        self._current_epoch = 0
        self._global_step = 0
        self._best_loss = float("inf")

        # Move model to device
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, train_data: Union[DataLoader, Dataset, List], epochs: int
    ) -> List[float]:
        """Train the model for a given number of epochs.

        Parameters
        ----------
        train_data : DataLoader, Dataset, or list of (input, target) tuples
            Training data in any supported format.
        epochs : int
            Number of epochs to train.

        Returns
        -------
        List[float]
            Average loss per epoch.
        """
        loader = self._adapt_data(train_data, shuffle=True)
        losses: List[float] = []

        # Set memory mode to TRAIN
        if self._memory_module is not None:
            self._memory_module.set_mode("TRAIN")

        logs: Dict[str, Any] = {"trainer": self}
        self._invoke_callbacks("on_train_start", logs=logs)

        for epoch in range(epochs):
            epoch_idx = self._current_epoch + epoch
            epoch_logs: Dict[str, Any] = {"epoch": epoch_idx, "trainer": self}
            self._invoke_callbacks("on_epoch_start", epoch=epoch_idx, logs=epoch_logs)

            epoch_loss = self._train_epoch(loader, epoch_idx)
            losses.append(epoch_loss)

            epoch_logs["loss"] = epoch_loss
            self._invoke_callbacks("on_epoch_end", epoch=epoch_idx, logs=epoch_logs)

            # Save memory per epoch if memory module exists
            if self._memory_module is not None:
                self._memory_module.save()

            # Check for early stopping flag
            if epoch_logs.get("stop_training", False):
                break

        self._current_epoch += len(losses)
        self._invoke_callbacks("on_train_end", logs=logs)

        return losses

    def evaluate(
        self, test_data: Union[DataLoader, Dataset, List]
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Parameters
        ----------
        test_data : DataLoader, Dataset, or list of (input, target) tuples
            Evaluation data in any supported format.

        Returns
        -------
        Dict[str, float]
            Metrics dictionary with at least a "loss" key.
        """
        loader = self._adapt_data(test_data, shuffle=False)

        # Save current mode and switch to INFERENCE
        previous_mode = None
        if self._memory_module is not None:
            previous_mode = self._memory_module.mode.value
            self._memory_module.set_mode("INFERENCE")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        try:
            with torch.no_grad():
                for inputs, targets in loader:
                    inputs = self._to_device(inputs)
                    targets = self._to_device(targets)

                    outputs = self.model(inputs)
                    # Handle 3D outputs for CrossEntropyLoss
                    if outputs.dim() == 3 and targets.dim() == 1:
                        loss = self.criterion(
                            outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
                        )
                    elif outputs.dim() == 3 and targets.dim() == 2:
                        loss = self.criterion(
                            outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
                        )
                    else:
                        loss = self.criterion(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
        finally:
            # Restore previous mode
            if self._memory_module is not None and previous_mode is not None:
                self._memory_module.set_mode(previous_mode)
            self.model.train()

        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run inference on input tensor.

        Parameters
        ----------
        input_data : torch.Tensor
            Input tensor for prediction.

        Returns
        -------
        torch.Tensor
            Raw model output tensor.
        """
        # Save current mode and switch to INFERENCE
        previous_mode = None
        if self._memory_module is not None:
            previous_mode = self._memory_module.mode.value
            self._memory_module.set_mode("INFERENCE")

        self.model.eval()

        try:
            with torch.no_grad():
                input_data = self._to_device(input_data)
                output = self.model(input_data)
        finally:
            # Restore previous mode
            if self._memory_module is not None and previous_mode is not None:
                self._memory_module.set_mode(previous_mode)
            self.model.train()

        return output

    def save(self, path: str) -> None:
        """Save complete model state to a directory.

        Saves model weights, optimizer state, configuration, training
        state, and AESG memory directory.

        Parameters
        ----------
        path : str
            Directory path to save the checkpoint.

        Raises
        ------
        AESGStorageError
            If the path is invalid or the save operation fails.
        """
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            raise AESGStorageError(
                f"Cannot create checkpoint directory '{path}': {e}"
            ) from e

        try:
            # Save model weights
            torch.save(
                self.model.state_dict(),
                os.path.join(path, "model_weights.pt"),
            )

            # Save optimizer state
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(path, "optimizer_state.pt"),
            )

            # Save config JSON (from memory module or basic trainer config)
            config_data = self._get_config_dict()
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(config_data, f, indent=2)

            # Save training state
            training_state = {
                "epoch": self._current_epoch,
                "global_step": self._global_step,
                "best_loss": self._best_loss,
                "device": self.device,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
            }
            with open(os.path.join(path, "training_state.json"), "w") as f:
                json.dump(training_state, f, indent=2)

            # Save memory directory (copy)
            if self._memory_module is not None:
                self._memory_module.save()
                memory_src = self._memory_module.directory
                memory_dst = os.path.join(path, "memory")
                if os.path.exists(memory_dst):
                    shutil.rmtree(memory_dst)
                if os.path.exists(memory_src):
                    shutil.copytree(memory_src, memory_dst)

        except (OSError, RuntimeError) as e:
            raise AESGStorageError(
                f"Failed to save checkpoint to '{path}': {e}"
            ) from e

    def load(self, path: str) -> None:
        """Load complete model state from a directory.

        Parameters
        ----------
        path : str
            Directory path containing the checkpoint.

        Raises
        ------
        AESGStorageError
            If the path is invalid or required files are missing.
        """
        if not os.path.isdir(path):
            raise AESGStorageError(
                f"Checkpoint directory does not exist: '{path}'"
            )

        weights_path = os.path.join(path, "model_weights.pt")
        if not os.path.isfile(weights_path):
            raise AESGStorageError(
                f"Model weights file missing: '{weights_path}'"
            )

        try:
            # Load model weights
            state_dict = torch.load(
                weights_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)

            # Load optimizer state
            optimizer_path = os.path.join(path, "optimizer_state.pt")
            if os.path.isfile(optimizer_path):
                opt_state = torch.load(
                    optimizer_path, map_location=self.device, weights_only=True
                )
                self.optimizer.load_state_dict(opt_state)

            # Load config
            config_path = os.path.join(path, "config.json")
            if os.path.isfile(config_path):
                with open(config_path, "r") as f:
                    json.load(f)  # Validate JSON, config is informational

            # Load training state
            state_path = os.path.join(path, "training_state.json")
            if os.path.isfile(state_path):
                with open(state_path, "r") as f:
                    training_state = json.load(f)
                self._current_epoch = training_state.get("epoch", 0)
                self._global_step = training_state.get("global_step", 0)
                self._best_loss = training_state.get(
                    "best_loss", float("inf")
                )

            # Load memory directory
            memory_path = os.path.join(path, "memory")
            if (
                self._memory_module is not None
                and os.path.isdir(memory_path)
            ):
                memory_dst = self._memory_module.directory
                # On Windows, mmap files are locked — copy files individually
                # instead of rmtree + copytree
                os.makedirs(memory_dst, exist_ok=True)
                for item in os.listdir(memory_path):
                    src_file = os.path.join(memory_path, item)
                    dst_file = os.path.join(memory_dst, item)
                    if os.path.isfile(src_file):
                        try:
                            shutil.copy2(src_file, dst_file)
                        except (PermissionError, OSError):
                            pass  # Skip locked mmap files on Windows

        except (OSError, RuntimeError) as e:
            raise AESGStorageError(
                f"Failed to load checkpoint from '{path}': {e}"
            ) from e

    def resume(self, checkpoint_path: str) -> int:
        """Resume training from a checkpoint.

        Loads all state (model, optimizer, training state, memory) and
        returns the epoch to resume from.

        Parameters
        ----------
        checkpoint_path : str
            Directory path containing the checkpoint.

        Returns
        -------
        int
            The epoch number to resume training from.

        Raises
        ------
        AESGStorageError
            If the checkpoint path is invalid or files are missing.
        """
        self.load(checkpoint_path)
        return self._current_epoch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch_idx: int) -> float:
        """Execute one training epoch.

        Parameters
        ----------
        loader : DataLoader
            Training data loader.
        epoch_idx : int
            Current epoch index (for logging).

        Returns
        -------
        float
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            batch_logs: Dict[str, Any] = {
                "batch": batch_idx,
                "epoch": epoch_idx,
                "trainer": self,
            }
            self._invoke_callbacks(
                "on_batch_start", batch=batch_idx, logs=batch_logs
            )

            inputs = self._to_device(inputs)
            targets = self._to_device(targets)

            # Reset memory state per batch
            if self._memory_module is not None:
                self._memory_module.reset_state()

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss — handle 3D outputs (B, S, C) for CrossEntropyLoss
            if outputs.dim() == 3 and targets.dim() == 1:
                # Classification with sequence: flatten to (B*S, C) vs (B*S,)
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
                )
            elif outputs.dim() == 3 and targets.dim() == 2:
                # Text model: (B, S, vocab) vs (B, S) targets
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
                )
            else:
                loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Topology update
            if self._memory_module is not None:
                self._memory_module.update_topology()

            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            self._global_step += 1

            batch_logs["loss"] = batch_loss
            self._invoke_callbacks(
                "on_batch_end", batch=batch_idx, logs=batch_logs
            )

        return total_loss / max(num_batches, 1)

    def _adapt_data(
        self, data: Any, shuffle: bool = False
    ) -> DataLoader:
        """Convert various data formats to a DataLoader.

        Parameters
        ----------
        data : DataLoader, Dataset, or list
            Input data in any supported format.
        shuffle : bool
            Whether to shuffle the data.

        Returns
        -------
        DataLoader
            A DataLoader wrapping the input data.

        Raises
        ------
        AESGTrainingError
            If the data format is unsupported or the dataset is empty.
        """
        if isinstance(data, DataLoader):
            # Validate non-empty
            if hasattr(data.dataset, "__len__") and len(data.dataset) == 0:
                raise AESGTrainingError(
                    "Empty dataset provided. Training requires at least one sample."
                )
            return data

        if isinstance(data, Dataset):
            if hasattr(data, "__len__") and len(data) == 0:
                raise AESGTrainingError(
                    "Empty dataset provided. Training requires at least one sample."
                )
            return DataLoader(
                data, batch_size=self.batch_size, shuffle=shuffle
            )

        if isinstance(data, list):
            if len(data) == 0:
                raise AESGTrainingError(
                    "Empty dataset provided. Training requires at least one sample."
                )
            dataset = _ListDataset(data)
            return DataLoader(
                dataset, batch_size=self.batch_size, shuffle=shuffle
            )

        raise AESGTrainingError(
            "Unsupported data format. Supported formats: "
            "DataLoader, Dataset, or list of (input, target) tuples."
        )

    def _find_memory(self, model: nn.Module):
        """Walk model.modules() and return the first AESGMemory found.

        Parameters
        ----------
        model : nn.Module
            The model to search.

        Returns
        -------
        Optional[AESGMemory]
            The first AESGMemory module found, or None.
        """
        # Import here to avoid circular imports at module level
        from aesg.memory.controller import AESGMemory

        for module in model.modules():
            if isinstance(module, AESGMemory):
                return module
        return None

    def _invoke_callbacks(self, method_name: str, **kwargs) -> None:
        """Invoke a callback method on all registered callbacks.

        Catches and logs exceptions from callbacks without interrupting
        training.

        Parameters
        ----------
        method_name : str
            Name of the callback method to invoke.
        **kwargs
            Arguments to pass to the callback method.
        """
        for callback in self.callbacks:
            try:
                method = getattr(callback, method_name, None)
                if method is not None:
                    method(**kwargs)
            except Exception as e:
                warnings.warn(
                    f"Callback {type(callback).__name__}.{method_name}() "
                    f"raised an exception: {e}",
                    stacklevel=2,
                )

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the trainer's device.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to move.

        Returns
        -------
        torch.Tensor
            Tensor on the target device.
        """
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return tensor

    def _get_config_dict(self) -> Dict[str, Any]:
        """Get configuration dictionary for serialization.

        Returns
        -------
        Dict[str, Any]
            Configuration data including AESG config if available.
        """
        config_data: Dict[str, Any] = {
            "trainer_version": "3.0.0",
            "device": self.device,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }

        if self._memory_module is not None:
            config_data["aesg_config"] = json.loads(
                self._memory_module.config.to_json()
            )

        return config_data
