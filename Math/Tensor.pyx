# cython: boundscheck=False, wraparound=False
import cython
import numpy as np
cimport numpy as np

cdef class Tensor:
    def __init__(self, data, shape=None):
        """
        Initializes a Tensor from nested list data with an optional shape.

        Parameters
        ----------
        data : nested list
            The input data for the tensor.
        shape : tuple, optional
            The shape to be used for the tensor. If None, it is inferred from data.
        """
        if shape is None:
            shape = self._infer_shape(data)

        flat_data = self._flatten(data)
        total_elements = self._compute_num_elements(shape)

        if total_elements != len(flat_data):
            raise ValueError("Shape does not match the number of elements in data.")

        self._shape = shape
        self._strides = self._compute_strides(shape)
        self._data = np.array(flat_data, dtype=np.float64)

    @property
    def shape(self):
        """Returns the shape of the tensor."""
        return self._shape

    @property
    def strides(self):
        """Returns the strides used in indexing."""
        return self._strides

    @property
    def data(self):
        """Returns the raw flat data."""
        return self._data

    cdef tuple _infer_shape(self, object data):
        """
        Recursively infers the shape of nested list data.

        Parameters
        ----------
        data : list
            A potentially nested list representing tensor data.

        Returns
        -------
        tuple
            Inferred shape of the tensor.
        """
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            return (len(data),) + self._infer_shape(data[0])
        return ()

    cdef list _flatten(self, object data):
        """
        Recursively flattens nested list data.

        Parameters
        ----------
        data : list
            A potentially nested list of tensor data.

        Returns
        -------
        list
            A flattened list of float values.
        """
        if isinstance(data, list):
            result = []
            for sub in data:
                result.extend(self._flatten(sub))
            return result
        return [data]

    cdef tuple _compute_strides(self, tuple shape):
        """
        Computes strides needed for indexing.

        Parameters
        ----------
        shape : tuple
            Tensor shape.

        Returns
        -------
        tuple
            Strides corresponding to each dimension.
        """
        cdef list strides = []
        cdef int product = 1
        for dim in reversed(shape):
            strides.append(product)
            product *= dim
        return tuple(reversed(strides))

    cdef int _compute_num_elements(self, tuple shape):
        """
        Computes total number of elements from the shape.

        Parameters
        ----------
        shape : tuple
            Tensor shape.

        Returns
        -------
        int
            Number of elements.
        """
        cdef int product = 1
        for dim in shape:
            product *= dim
        return product

    cdef void _validate_indices(self, tuple indices):
        """
        Validates that indices are within the valid range for each dimension.

        Parameters
        ----------
        indices : tuple
            Index tuple to validate.

        Raises
        ------
        IndexError
            If any index is out of bounds.
        """
        cdef int i
        if len(indices) != len(self._shape):
            raise IndexError(f"Expected {len(self._shape)} indices but got {len(indices)}.")
        for i in range(len(indices)):
            if indices[i] < 0 or indices[i] >= self._shape[i]:
                raise IndexError(f"Index {indices} is out of bounds for shape {self._shape}.")

    cpdef float get(self, tuple indices):
        """
        Retrieve an element by its multi-dimensional index.

        Parameters
        ----------
        indices : tuple
            Multi-dimensional index for accessing the tensor value.

        Returns
        -------
        float
            The value at the specified index.
        """
        self._validate_indices(indices)

        cdef Py_ssize_t flat_index = 0
        cdef int i
        for i in range(len(indices)):
            flat_index += indices[i] * self._strides[i]
        return self._data[flat_index]

    cpdef void set(self, tuple indices, float value):
        """
        Set a value at a specific index in the tensor.

        Parameters
        ----------
        indices : tuple
            Multi-dimensional index for accessing the tensor.
        value : float
            The value to be assigned at the specified index.
        """
        self._validate_indices(indices)

        cdef Py_ssize_t flat_index = 0
        cdef int i
        for i in range(len(indices)):
            flat_index += indices[i] * self._strides[i]
        self._data[flat_index] = value

    cpdef Tensor reshape(self, tuple new_shape):
        """
        Reshapes the tensor into a new shape without changing the data.

        Parameters
        ----------
        new_shape : tuple
            New desired shape.

        Returns
        -------
        Tensor
            Tensor with new shape.

        Raises
        ------
        ValueError
            If total number of elements mismatches.
        """
        if self._compute_num_elements(new_shape) != self._compute_num_elements(self._shape):
            raise ValueError("Total number of elements must remain the same.")
        return Tensor(np.asarray(self._data).tolist(), new_shape)

    cpdef Tensor transpose(self, tuple axes=None):
        """
        Transposes the tensor according to the given axes.

        Parameters
        ----------
        axes : tuple, optional
            The order of axes. If None, the axes are reversed.

        Returns
        -------
        Tensor
            A new tensor with transposed data.
        """
        cdef int ndim = len(self._shape)

        if axes is None:
            axes = tuple(range(ndim - 1, -1, -1))
        else:
            if sorted(axes) != list(range(ndim)):
                raise ValueError("Invalid transpose axes.")

        cdef list shape_list = []
        cdef int axis
        for axis in axes:
            shape_list.append(self._shape[axis])
        cdef tuple new_shape = tuple(shape_list)

        cdef list flattened_data = self._transpose_flattened_data(axes, new_shape)

        return Tensor(flattened_data, new_shape)

    cdef list _transpose_flattened_data(self, tuple axes, tuple new_shape):
        """
        Rearranges the flattened data for transposition.

        Parameters
        ----------
        axes : tuple
            Tuple representing the order of axes.
        new_shape : tuple
            Tuple representing the new shape.

        Returns
        -------
        list
            Flattened list of transposed data.
        """
        cdef tuple new_strides = self._compute_strides(new_shape)
        cdef list flattened_data = []
        cdef int i, dim
        cdef tuple new_indices
        cdef list original_indices

        for i in range(self._compute_num_elements(new_shape)):
            new_indices = self._unflatten_index(i, new_strides)
            original_indices = [new_indices[axes.index(dim)] for dim in range(len(self._shape))]
            flattened_data.append(self.get(tuple(original_indices)))

        return flattened_data

    cdef tuple _unflatten_index(self, int flat_index, tuple strides):
        """
        Converts flat index to multi-dimensional index using strides.

        Parameters
        ----------
        flat_index : int
            Flat index into the tensor.
        strides : tuple
            Strides for each dimension.

        Returns
        -------
        tuple
            Multi-dimensional index.
        """
        cdef list indices = []
        for stride in strides:
            indices.append(flat_index // stride)
            flat_index %= stride
        return tuple(indices)

    cdef tuple _broadcast_shape(self, tuple shape1, tuple shape2):
        """
        Calculates broadcasted shape from two input shapes.

        Parameters
        ----------
        shape1 : tuple
            Shape of the first tensor.
        shape2 : tuple
            Shape of the second tensor.

        Returns
        -------
        tuple
            Broadcasted shape.
        """
        cdef list r1 = list(reversed(shape1))
        cdef list r2 = list(reversed(shape2))
        cdef list result = []
        for i in range(min(len(r1), len(r2))):
            d1 = r1[i]
            d2 = r2[i]
            if d1 == d2:
                result.append(d1)
            elif d1 == 1 or d2 == 1:
                result.append(max(d1, d2))
            else:
                raise ValueError(f"Shapes {shape1} and {shape2} not broadcastable")
        result.extend(r1[len(r2):])
        result.extend(r2[len(r1):])
        return tuple(reversed(result))

    cpdef Tensor broadcast_to(self, tuple target_shape):
        """
        Broadcasts the tensor to a new shape.

        Parameters
        ----------
        target_shape : tuple
            Target shape to broadcast the current tensor to.

        Returns
        -------
        Tensor
            New tensor with broadcasted data.

        Raises
        ------
        ValueError
            If broadcasting is not possible.
        """
        cdef int i, j
        cdef int rank = len(target_shape)
        cdef int size = self._compute_num_elements(target_shape)
        cdef double[:] new_data = np.empty(size, dtype=np.float64)

        cdef tuple expanded_shape = (1,) * (rank - len(self._shape)) + self._shape
        for i in range(rank):
            if not (expanded_shape[i] == target_shape[i] or expanded_shape[i] == 1):
                raise ValueError(f"Cannot broadcast shape {self._shape} to {target_shape}")

        cdef tuple target_strides = self._compute_strides(target_shape)
        cdef tuple indices
        cdef list temp
        for i in range(size):
            indices = self._unflatten_index(i, target_strides)
            temp = []
            for j in range(rank):
                temp.append(indices[j] if expanded_shape[j] > 1 else 0)
            new_data[i] = self.get(tuple(temp))

        return Tensor(np.asarray(new_data).tolist(), target_shape)

    def __add__(self, Tensor other):
        """
        Adds two tensors element-wise with broadcasting support.
        """
        cdef tuple shape = self._broadcast_shape(self._shape, other._shape)
        cdef Tensor t1 = self.broadcast_to(shape)
        cdef Tensor t2 = other.broadcast_to(shape)
        cdef int size = self._compute_num_elements(shape)
        cdef double[:] result_data = np.empty(size, dtype=np.float64)
        cdef int i
        cdef tuple idx
        for i in range(size):
            idx = self._unflatten_index(i, t1._strides)
            result_data[i] = t1.get(idx) + t2.get(idx)
        return Tensor(np.asarray(result_data).tolist(), shape)

    def __sub__(self, Tensor other):
        """
        Subtracts the other tensor from self, element-wise, with broadcasting.
        """
        cdef tuple shape = self._broadcast_shape(self._shape, other._shape)
        cdef Tensor t1 = self.broadcast_to(shape)
        cdef Tensor t2 = other.broadcast_to(shape)
        cdef int size = self._compute_num_elements(shape)
        cdef double[:] result_data = np.empty(size, dtype=np.float64)
        cdef int i
        cdef tuple idx
        for i in range(size):
            idx = self._unflatten_index(i, t1._strides)
            result_data[i] = t1.get(idx) - t2.get(idx)
        return Tensor(np.asarray(result_data).tolist(), shape)

    def __mul__(self, Tensor other):
        """
        Multiplies two tensors element-wise with broadcasting support.
        """
        cdef tuple shape = self._broadcast_shape(self._shape, other._shape)
        cdef Tensor t1 = self.broadcast_to(shape)
        cdef Tensor t2 = other.broadcast_to(shape)
        cdef int size = self._compute_num_elements(shape)
        cdef double[:] result_data = np.empty(size, dtype=np.float64)
        cdef int i
        cdef tuple idx
        for i in range(size):
            idx = self._unflatten_index(i, t1._strides)
            result_data[i] = t1.get(idx) * t2.get(idx)
        return Tensor(np.asarray(result_data).tolist(), shape)

    cpdef Tensor dot(self, Tensor other):
        """
        Computes the dot product between two 2D tensors (matrix multiplication).

        Parameters
        ----------
        other : Tensor
            Another tensor to compute dot product with.

        Returns
        -------
        Tensor
            Result of the dot product.

        Raises
        ------
        ValueError
            If shapes are not aligned for matrix multiplication.
        """
        cdef int ndim_self = len(self._shape)
        cdef int ndim_other = len(other._shape)
        if self._shape[ndim_self - 1] != other._shape[ndim_other - 2]:
            raise ValueError(f"Shapes {self._shape} and {other._shape} are not aligned for dot product.")

        cdef int m = self._shape[0]
        cdef int n = self._shape[1]
        cdef int p = other._shape[1]

        cdef double[:] result_data = np.empty(m * p, dtype=np.float64)
        cdef int i, j, k
        cdef double acc

        for i in range(m):
            for j in range(p):
                acc = 0.0
                for k in range(n):
                    acc += self.get((i, k)) * other.get((k, j))
                result_data[i * p + j] = acc

        return Tensor(np.asarray(result_data).tolist(), (m, p))

    cpdef Tensor partial(self, tuple start_indices, tuple end_indices):
        """
        Extracts a sub-tensor from the given index ranges.

        Parameters
        ----------
        start_indices : tuple
            Start indices for each axis.
        end_indices : tuple
            End indices (exclusive) for each axis.

        Returns
        -------
        Tensor
            Extracted sub-tensor.

        Raises
        ------
        ValueError
            If the indices do not match tensor dimensions.
        """
        if len(start_indices) != len(self._shape) or len(end_indices) != len(self._shape):
            raise ValueError("start_indices and end_indices must match the number of dimensions.")

        cdef tuple new_shape = tuple([end - start for start, end in zip(start_indices, end_indices)])
        cdef int size = self._compute_num_elements(new_shape)
        cdef double[:] sub_data = np.empty(size, dtype=np.float64)

        cdef tuple sub_strides = self._compute_strides(new_shape)
        cdef int i
        cdef tuple sub_indices, original_indices

        for i in range(size):
            sub_indices = self._unflatten_index(i, sub_strides)
            original_indices = tuple([start + offset for start, offset in zip(start_indices, sub_indices)])
            sub_data[i] = self.get(original_indices)

        return Tensor(np.asarray(sub_data).tolist(), new_shape)

    def __repr__(self):
        """
        Returns a string representation of the tensor showing shape and data.
        """
        cdef list data_list = list(self._data)
        cdef int dim = len(self._shape)
        cdef str result = f"Tensor(shape={self._shape}, data="
        if dim == 1:
            result += str(data_list)
        elif dim == 2:
            stride = self._shape[1]
            result += str([data_list[i * stride:(i + 1) * stride] for i in range(self._shape[0])])
        else:
            result += "<nd tensor>"
        result += ")"
        return result

