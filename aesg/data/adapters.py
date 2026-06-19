"""
AESG Data Adapters.

Public utilities for adapting various data formats into PyTorch DataLoaders
suitable for use with AESGTrainer or standalone training loops.
"""

from typing import Any, List, Union

from torch.utils.data import DataLoader, Dataset

from aesg.exceptions import AESGTrainingError


class ListDataset(Dataset):
    """Wraps a list of (input, target) tuples as a PyTorch Dataset.

    Each item in the list must be a tuple or list of exactly two elements:
    (input, target).

    Parameters
    ----------
    data : List
        List of (input, target) tuples.

    Raises
    ------
    AESGTrainingError
        If an item is not a 2-element tuple/list on access.

    Examples
    --------
    >>> ds = ListDataset([(torch.tensor([1.0]), torch.tensor([0]))])
    >>> len(ds)
    1
    >>> ds[0]
    (tensor([1.0]), tensor([0]))
    """

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


def adapt_data(
    data: Union[DataLoader, Dataset, List],
    batch_size: int = 32,
    shuffle: bool = False,
) -> DataLoader:
    """Convert various data formats to a PyTorch DataLoader.

    Supports DataLoader (passthrough), Dataset, and list of (input, target)
    tuples. Raises on unsupported formats or empty datasets.

    Parameters
    ----------
    data : DataLoader, Dataset, or list
        Input data in any supported format.
    batch_size : int
        Batch size for the resulting DataLoader (ignored if data is
        already a DataLoader).
    shuffle : bool
        Whether to shuffle the data (ignored if data is already a
        DataLoader).

    Returns
    -------
    DataLoader
        A DataLoader wrapping the input data.

    Raises
    ------
    AESGTrainingError
        If the data format is unsupported or the dataset is empty.

    Examples
    --------
    >>> loader = adapt_data([(x, y) for x, y in zip(inputs, targets)])
    >>> for batch_x, batch_y in loader:
    ...     pass
    """
    if isinstance(data, DataLoader):
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
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    if isinstance(data, list):
        if len(data) == 0:
            raise AESGTrainingError(
                "Empty dataset provided. Training requires at least one sample."
            )
        dataset = ListDataset(data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    raise AESGTrainingError(
        "Unsupported data format. Supported formats: "
        "DataLoader, Dataset, or list of (input, target) tuples."
    )
