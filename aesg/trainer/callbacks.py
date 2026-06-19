"""
AESG Training Callbacks.

Extensible callback interface for training lifecycle hooks. Includes
built-in callbacks for early stopping, checkpointing, TensorBoard, and
Weights & Biases logging.
"""

import os
import warnings
from typing import Any, Dict, Optional


class Callback:
    """Base callback interface. Override methods as needed.

    All lifecycle methods default to no-op. Subclass and override
    only the methods relevant to your use case.
    """

    def on_train_start(self, logs: Dict[str, Any]) -> None:
        pass

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        pass

    def on_epoch_start(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_batch_start(self, batch: int, logs: Dict[str, Any]) -> None:
        pass

    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        pass


class EarlyStopping(Callback):
    """Stops training when the monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    monitor : str
        Metric name to monitor in epoch logs.

    Examples
    --------
    >>> from aesg.trainer.callbacks import EarlyStopping
    >>> cb = EarlyStopping(patience=3, monitor="loss")
    >>> trainer = AESGTrainer(model, callbacks=[cb])
    """

    def __init__(self, patience: int = 5, monitor: str = "loss"):
        self.patience = patience
        self.monitor = monitor
        self._counter = 0
        self._best = float("inf")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        current = logs.get(self.monitor, float("inf"))
        if current < self._best:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logs["stop_training"] = True


class Checkpoint(Callback):
    """Saves model checkpoints at regular epoch intervals.

    Parameters
    ----------
    save_every : int
        Save a checkpoint every N epochs.
    path : str
        Base directory for checkpoint storage.

    Examples
    --------
    >>> from aesg.trainer.callbacks import Checkpoint
    >>> cb = Checkpoint(save_every=2, path="./my_checkpoints")
    >>> trainer = AESGTrainer(model, callbacks=[cb])
    """

    def __init__(self, save_every: int = 1, path: str = "./checkpoints"):
        self.save_every = save_every
        self.path = path

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if (epoch + 1) % self.save_every == 0:
            trainer = logs.get("trainer")
            if trainer:
                save_path = os.path.join(self.path, f"epoch_{epoch + 1}")
                trainer.save(save_path)


class TensorBoardCallback(Callback):
    """Logs training metrics and AESG memory stats to TensorBoard.

    Attempts to import ``torch.utils.tensorboard.SummaryWriter``.
    If TensorBoard is not installed, emits a warning and operates
    as a no-op.

    Parameters
    ----------
    log_dir : str
        Directory for TensorBoard log files.

    Examples
    --------
    >>> from aesg.trainer.callbacks import TensorBoardCallback
    >>> cb = TensorBoardCallback(log_dir="./runs/experiment_1")
    >>> trainer = AESGTrainer(model, callbacks=[cb])
    """

    def __init__(self, log_dir: str = "./runs"):
        self.log_dir = log_dir
        self._writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            warnings.warn(
                "TensorBoard not available. Install tensorboard to enable "
                "TensorBoardCallback logging: pip install tensorboard",
                stacklevel=2,
            )

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self._writer is None:
            return

        # Log loss
        loss = logs.get("loss")
        if loss is not None:
            self._writer.add_scalar("loss", loss, epoch)

        # Log AESG memory stats if available
        trainer = logs.get("trainer")
        if trainer and hasattr(trainer, "_memory_module") and trainer._memory_module is not None:
            memory = trainer._memory_module
            if hasattr(memory, "node_count"):
                self._writer.add_scalar(
                    "memory/node_count", memory.node_count, epoch
                )
            if hasattr(memory, "active_nodes"):
                self._writer.add_scalar(
                    "memory/active_nodes", memory.active_nodes, epoch
                )

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self._writer is not None:
            self._writer.close()


class WandBCallback(Callback):
    """Logs training metrics and AESG memory stats to Weights & Biases.

    Attempts to import ``wandb``. If wandb is not installed, emits a
    warning and operates as a no-op.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : Optional[str]
        Optional name for this run. If None, wandb generates one.

    Examples
    --------
    >>> from aesg.trainer.callbacks import WandBCallback
    >>> cb = WandBCallback(project="my-aesg-project", run_name="exp_01")
    >>> trainer = AESGTrainer(model, callbacks=[cb])
    """

    def __init__(self, project: str = "aesg", run_name: Optional[str] = None):
        self.project = project
        self.run_name = run_name
        self._wandb = None

        try:
            import wandb

            self._wandb = wandb
        except ImportError:
            warnings.warn(
                "wandb not available. Install wandb to enable "
                "WandBCallback logging: pip install wandb",
                stacklevel=2,
            )

    def on_train_start(self, logs: Dict[str, Any]) -> None:
        if self._wandb is None:
            return

        self._wandb.init(project=self.project, name=self.run_name)

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self._wandb is None:
            return

        log_data: Dict[str, Any] = {"epoch": epoch}

        # Log loss
        loss = logs.get("loss")
        if loss is not None:
            log_data["loss"] = loss

        # Log AESG memory stats if available
        trainer = logs.get("trainer")
        if trainer and hasattr(trainer, "_memory_module") and trainer._memory_module is not None:
            memory = trainer._memory_module
            if hasattr(memory, "node_count"):
                log_data["memory/node_count"] = memory.node_count
            if hasattr(memory, "active_nodes"):
                log_data["memory/active_nodes"] = memory.active_nodes

        self._wandb.log(log_data)

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self._wandb is None:
            return

        self._wandb.finish()
