from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
    ----
        index : index tuple of ints
        strides : tensor strides

    Returns:
    -------
        Position in storage

    """
    position = 0
    for i, stride in zip(index, strides):
        position += i * stride
    return position


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an `ordinal` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
    ----
        ordinal: ordinal position to convert.
        shape : tensor shape.
        out_index : return index corresponding to position.

    """
    # TODO: Implement for Task 2.1.
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = ordinal % shape[i]
        ordinal //= shape[i]


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Broadcast an index to match the shape of a larger tensor.

    Args:
    ----
        big_index (Index): The index to broadcast.
        big_shape (Shape): The shape of the larger tensor.
        shape (Shape): The shape to match.
        out_index (OutIndex): The output index after broadcasting.

    """
    # Ensure that the shapes are compatible
    if len(big_shape) < len(shape):
        raise ValueError("big_shape must be at least as long as shape")

    # Iterate through dimensions from last to first
    for i in range(len(shape) - 1, -1, -1):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i]

    # Fill any additional dimensions in big_shape with 0
    zeros_to_prepend = np.zeros(len(big_shape) - len(shape), dtype=out_index.dtype)
    out_index = np.concatenate((zeros_to_prepend, out_index))


def test_broadcast_index() -> None:
    """Test the broadcast_index function.

    Returns
    -------
        None

    """
    big_shape = np.array([2, 3, 4], dtype=np.int32)
    shape = np.array([3, 1], dtype=np.int32)
    big_index = np.array([1, 2, 3], dtype=np.int32)
    out_index = np.zeros(len(shape), dtype=np.int32)
    broadcast_index(big_index, big_shape, shape, out_index)
    # print(f"Output Index: {out_index}")  # Should output: [2, 0]


def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape.

    Args:
    ----
        shape_a (UserShape): The first shape to broadcast.
        shape_b (UserShape): The second shape to broadcast.

    Returns:
    -------
        UserShape: The broadcasted shape.

    Raises:
    ------
        ValueError: If the shapes cannot be broadcasted.

    """
    result_shape = []
    len1, len2 = len(shape_a), len(shape_b)
    for i in range(max(len1, len2)):
        dim1 = shape_a[len1 - 1 - i] if i < len1 else 1
        dim2 = shape_b[len2 - 1 - i] if i < len2 else 1
        if dim1 == 1 or dim2 == 1 or dim1 == dim2:
            result_shape.append(max(dim1, dim2))
        else:
            raise IndexingError(f"Shapes {shape_a} and {shape_b} cannot be broadcasted")
    return tuple(reversed(result_shape))


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return a contiguous stride for a shape"""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """Class representing the core data structure for tensors."""

    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert to cuda"""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns
        -------
            bool : True if contiguous

        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to create a new union shape.

        Args:
        ----
            shape_a (UserShape): The first shape to broadcast.
            shape_b (UserShape): The second shape to broadcast.

        Returns:
        -------
            UserShape: The broadcasted shape.

        Raises:
        ------
            ValueError: If the shapes cannot be broadcasted.

        """
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Get the position in storage for a given index.

        Args:
        ----
            index (Union[int, UserIndex]): The index to convert.

        Returns:
        -------
            int: The position in storage.

        Raises:
        ------
            IndexingError: If the index is out of range or invalid.

        """
        # print(f"type of index is {type(index)}")
        # print(f"user index is {index}")
        if isinstance(index, int):
            aindex: Index = array([index])
        else:  # if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Generate all valid indices for the tensor.

        Returns
        -------
            Iterable[UserIndex]: An iterable of valid indices.

        """
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Get a random valid index.

        Returns
        -------
            UserIndex: A random valid index.

        """
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Get the value at a specific index.

        Args:
        ----
            key (UserIndex): The index to retrieve the value from.

        Returns:
        -------
            float: The value at the specified index.

        """
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set the value at a specific index.

        Args:
        ----
            key (UserIndex): The index to set the value at.
            val (float): The value to set.

        """
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the dimensions of the tensor.

        Args:
        ----
            *order: A permutation of the dimensions.

        Returns:
        -------
            TensorData: A new TensorData with the same storage and a new dimension order.

        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides = tuple(self._strides[i] for i in order)
        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        """Convert to string"""
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
