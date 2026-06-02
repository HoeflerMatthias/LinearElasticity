import tensorflow as tf


class DataSet:
    """Wraps a tensor with a name and optional mini-batch support.

    Parameters
    ----------
    data : tensor-like
        The full dataset tensor.
    name : str
        Key used by ``DataCollection.current_batch``.
    batch_size : int or None
        If ``None``, ``get_batch`` returns the full data.
    """

    def __init__(self, data, name, batch_size=None):
        self.data = data if isinstance(data, tf.Tensor) else tf.constant(data, dtype=tf.float64)
        self.name = name
        self.batch_size = batch_size
        self._index = 0
        self._current_batch = None

    def advance(self):
        """Advance the internal pointer and cache the next batch."""
        n = self.data.shape[0]
        if self.batch_size is None or self.batch_size >= n:
            self._current_batch = self.data
            return

        end = self._index + self.batch_size
        if end <= n:
            self._current_batch = self.data[self._index:end]
        else:
            self._current_batch = tf.concat(
                [self.data[self._index:], self.data[:end - n]], axis=0
            )
        self._index = end % n

    def get_batch(self):
        if self._current_batch is None:
            self.advance()
        return self._current_batch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self._index = 0
        self._current_batch = None


class DataCollection:
    """Collection of :class:`DataSet` instances — provides batched data dicts.

    Parameters
    ----------
    datasets : list[DataSet]
        Constituent datasets.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self._current_batch = None

    def advance(self):
        for ds in self.datasets:
            ds.advance()
        self._current_batch = {ds.name: ds.get_batch() for ds in self.datasets}

    @property
    def current_batch(self):
        if self._current_batch is None:
            self.advance()
        return self._current_batch

    def set_batch_size(self, batch_size):
        for ds in self.datasets:
            ds.set_batch_size(batch_size)
        self._current_batch = None
