# cython: boundscheck=False
import array

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
            shape = self.__infer_shape(data)
        flat_data = self.__flatten(data)
        total_elements = self.__compute_num_elements(shape)
        if total_elements != len(flat_data):
            raise ValueError("Shape does not match the number of elements in data.")
        self.__shape = shape
        self.__strides = self.__compute_strides(shape)
        data_array = array.array('d', flat_data)
        self.__data = data_array

    def getShape(self):
        """Returns the shape of the tensor."""
        return self.__shape

    def getData(self):
        """Returns the raw flat data."""
        return self.__data

    cdef tuple __infer_shape(self, object data):
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
            return (len(data),) + self.__infer_shape(data[0])
        return ()

    cdef list __flatten(self, object data):
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
                result.extend(self.__flatten(sub))
            return result
        return [data]

    cdef tuple __compute_strides(self, tuple shape):
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

    cdef int __compute_num_elements(self, tuple shape):
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

    cdef void __validate_indices(self, tuple indices):
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
        if len(indices) != len(self.__shape):
            raise IndexError(f"Expected {len(self.__shape)} indices but got {len(indices)}.")
        for i in range(len(indices)):
            if indices[i] < 0 or indices[i] >= self.__shape[i]:
                raise IndexError(f"Index {indices} is out of bounds for shape {self.__shape}.")

    cpdef Tensor concat(self, Tensor other, int dimension):
        """
        Concatenates two tensors into a one.
        :param other: 2nd tensor for concatenation.
        :param dimension: to concatenate.
        :return: Concatenated Tensor.
        """
        cdef int start_index = 1
        cdef int end_index1 = 1
        cdef int end_index2 = 1
        cdef int i, j
        cdef list new_shape, new_list
        for i in range(len(self.__shape)):
            if i >= dimension:
                end_index1 *= self.__shape[i]
                end_index2 *= other.__shape[i]
            else:
                start_index *= self.__shape[i]
        new_shape = []
        for i in range(len(self.__shape)):
            if i == dimension:
                new_shape.append(self.__shape[i] + other.__shape[i])
            else:
                new_shape.append(self.__shape[i])
        new_list = []
        for i in range(start_index):
            for j in range(end_index1):
                new_list.append(self.__data[i * end_index1 + j])
            for j in range(end_index2):
                new_list.append(self.__data[i * end_index2 + j])
        return Tensor(new_list, tuple(new_shape))

    cpdef Tensor get(self, tuple dimensions):
        """
        Returns the subtensor taking the given dimensions.
        :param dimensions: Given dimensions
        :return: a subTensor
        """
        cdef list new_shape = self.__shape[len(dimensions):len(self.__shape)]
        cdef int i = 0
        cdef int start = 0
        cdef int end = len(self.__data)
        cdef int parts
        while i < len(dimensions):
            parts = (end - start) // self.__shape[i]
            start += parts * dimensions[i]
            end = start + parts
            i = i + 1
        return Tensor(self.__data[start:end], tuple(new_shape))

    cpdef float getValue(self, tuple indices):
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
        self.__validate_indices(indices)
        cdef Py_ssize_t flat_index = 0
        cdef int i
        for i in range(len(indices)):
            flat_index += indices[i] * self.__strides[i]
        return self.__data[flat_index]

    cpdef void setValue(self, tuple indices, float value):
        """
        Set a value at a specific index in the tensor.

        Parameters
        ----------
        indices : tuple
            Multi-dimensional index for accessing the tensor.
        value : float
            The value to be assigned at the specified index.
        """
        self.__validate_indices(indices)

        cdef Py_ssize_t flat_index = 0
        cdef int i
        for i in range(len(indices)):
            flat_index += indices[i] * self.__strides[i]
        self.__data[flat_index] = value

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
        if self.__compute_num_elements(new_shape) != self.__compute_num_elements(self.__shape):
            raise ValueError("Total number of elements must remain the same.")
        # Convert self._data to a flat list before passing to Tensor
        return Tensor(list(self.__data), new_shape)

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
        cdef int ndim = len(self.__shape)
        if axes is None:
            axes = tuple(range(ndim - 1, -1, -1))
        else:
            if sorted(axes) != list(range(ndim)):
                raise ValueError("Invalid transpose axes.")
        cdef list shape_list = []
        cdef int axis
        for axis in axes:
            shape_list.append(self.__shape[axis])
        cdef tuple new_shape = tuple(shape_list)
        cdef list flattened_data = self.__transpose_flattened_data(axes, new_shape)
        return Tensor(flattened_data, new_shape)

    cdef list __transpose_flattened_data(self, tuple axes, tuple new_shape):
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
        cdef tuple new_strides = self.__compute_strides(new_shape)
        cdef list flattened_data = []
        cdef int i, dim
        cdef tuple new_indices
        cdef list original_indices
        for i in range(self.__compute_num_elements(new_shape)):
            new_indices = self.__unflatten_index(i, new_strides)
            original_indices = [new_indices[axes.index(dim)] for dim in range(len(self.__shape))]
            flattened_data.append(self.getValue(tuple(original_indices)))
        return flattened_data

    cdef tuple __unflatten_index(self, int flat_index, tuple strides):
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

    cdef tuple __broadcast_shape(self, tuple shape1, tuple shape2):
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
        cdef int i, j, rank, size
        cdef tuple expanded, targ_strides, strides
        cdef list new_data
        cdef tuple idx
        cdef list orig_idx
        rank = len(target_shape)
        size = self.__compute_num_elements(target_shape)
        expanded = (1,) * (rank - len(self.__shape)) + self.__shape
        for i in range(rank):
            if not (expanded[i] == target_shape[i] or expanded[i] == 1):
                raise ValueError(f"Cannot broadcast shape {self.__shape} to {target_shape}")
        targ_strides = self.__compute_strides(target_shape)
        strides = self.__strides
        new_data = [0.0] * size
        for i in range(size):
            idx = self.__unflatten_index(i, targ_strides)
            orig_idx = []
            for j in range(rank):
                if expanded[j] > 1:
                    orig_idx.append(idx[j])
                else:
                    orig_idx.append(0)
            # Only use the last len(self._shape) indices for get()
            get_indices = tuple(orig_idx[-len(self.__shape):])
            new_data[i] = self.getValue(get_indices)
        return Tensor(new_data, target_shape)

    cpdef Tensor multiply(self, Tensor other):
        """
        Performs matrix multiplication (batched if necessary).

        For tensors of shape (..., M, K) and (..., K, N), returns (..., M, N).

        :param other: Tensor with shape compatible for matrix multiplication.
        :return: Tensor resulting from matrix multiplication.
        """
        if self.__shape[len(self.__shape) - 1] != other.__shape[len(other.__shape) - 2]:
            raise ValueError(f"Shapes {self.__shape} and {other.__shape} are not aligned for multiplication.")
        cdef tuple batch_shape = self.__shape[:-2]
        cdef int m = self.__shape[len(self.__shape) - 2]
        cdef int k1 = self.__shape[len(self.__shape) - 1]
        cdef int k2 = other.__shape[len(other.__shape) - 2]
        cdef int n = other.__shape[len(other.__shape) - 1]
        if k1 != k2:
            raise ValueError("Inner dimensions must match for matrix multiplication.")
        # Broadcasting batch shape if necessary
        cdef tuple broadcast_shape
        cdef Tensor self_broadcasted
        cdef Tensor other_broadcasted
        if batch_shape != other.__shape[:-2]:
            broadcast_shape = self.__broadcast_shape(self.__shape[:-2], other.__shape[:-2])
            self_broadcasted = self.broadcast_to(broadcast_shape + (m, k1))
            other_broadcasted = other.broadcast_to(broadcast_shape + (k2, n))
        else:
            broadcast_shape = batch_shape
            self_broadcasted = self
            other_broadcasted = other
        cdef tuple result_shape = broadcast_shape + (m, n)
        cdef list result_data = []
        cdef int num_elements = self.__compute_num_elements(result_shape)
        cdef tuple result_strides = self.__compute_strides(result_shape)
        cdef int i, k
        cdef tuple indices, batch_idx
        cdef int row, col
        cdef double sum_result
        cdef tuple a_idx, b_idx
        for i in range(num_elements):
            indices = self.__unflatten_index(i, result_strides)
            batch_idx = indices[:-2]
            row = indices[len(indices)-2]
            col = indices[len(indices)-1]
            sum_result = 0.0
            for k in range(k1):
                a_idx = batch_idx + (row, k)
                b_idx = batch_idx + (k, col)
                sum_result += self_broadcasted.getValue(a_idx) * other_broadcasted.getValue(b_idx)
            result_data.append(sum_result)
        return Tensor(result_data, result_shape)

    cpdef Tensor add(self, Tensor other):
        """
        Adds two tensors element-wise with broadcasting.

        :param other: The other tensor to add.
        :return: New tensor with the result of the addition.
        """
        cdef tuple broadcast_shape = self.__broadcast_shape(self.__shape, other.__shape)
        cdef Tensor tensor1 = self.broadcast_to(broadcast_shape)
        cdef Tensor tensor2 = other.broadcast_to(broadcast_shape)
        cdef int num_elements = self.__compute_num_elements(broadcast_shape)
        cdef list result_data = []
        cdef int i
        for i in range(num_elements):
            result_data.append(tensor1.__data[i] + tensor2.__data[i])
        
        return Tensor(result_data, broadcast_shape)

    cpdef Tensor subtract(self, Tensor other):
        """
        Subtracts one tensor from another element-wise with broadcasting.

        :param other: The other tensor to subtract.
        :return: New tensor with the result of the subtraction.
        """
        cdef tuple broadcast_shape = self.__broadcast_shape(self.__shape, other.__shape)
        cdef Tensor tensor1 = self.broadcast_to(broadcast_shape)
        cdef Tensor tensor2 = other.broadcast_to(broadcast_shape)
        cdef int num_elements = self.__compute_num_elements(broadcast_shape)
        cdef list result_data = []
        cdef int i
        for i in range(num_elements):
            result_data.append(tensor1.__data[i] - tensor2.__data[i])
        return Tensor(result_data, broadcast_shape)

    cpdef Tensor hadamardProduct(self, Tensor other):
        """
        Multiplies two tensors element-wise with broadcasting.

        :param other: The other tensor to multiply.
        :return: New tensor with the result of the multiplication.
        """
        cdef tuple broadcast_shape = self.__broadcast_shape(self.__shape, other.__shape)
        cdef Tensor tensor1 = self.broadcast_to(broadcast_shape)
        cdef Tensor tensor2 = other.broadcast_to(broadcast_shape)
        cdef int num_elements = self.__compute_num_elements(broadcast_shape)
        cdef list result_data = []
        cdef int i
        for i in range(num_elements):
            result_data.append(tensor1.__data[i] * tensor2.__data[i])
        return Tensor(result_data, broadcast_shape)

    cpdef Tensor partial(self, tuple start_indices, tuple end_indices):
        """
        Extracts a sub-tensor from the given start indices to the end indices.

        :param start_indices: Tuple specifying the start indices for each dimension.
        :param end_indices: Tuple specifying the end indices (exclusive) for each dimension.
        :return: A new Tensor containing the extracted sub-tensor.
        """
        if len(start_indices) != len(self.__shape) or len(end_indices) != len(self.__shape):
            raise ValueError("start_indices and end_indices must match the number of dimensions.")
        # Compute the new shape of the extracted sub-tensor
        cdef list new_shape_list = []
        cdef int i
        for i in range(len(start_indices)):
            new_shape_list.append(end_indices[i] - start_indices[i])
        cdef tuple new_shape = tuple(new_shape_list)
        # Extract data from the original tensor
        cdef list sub_data = []
        cdef int num_elements = self.__compute_num_elements(new_shape)
        cdef tuple new_strides = self.__compute_strides(new_shape)
        cdef tuple sub_indices, original_indices
        cdef list orig_indices_list
        cdef int j
        for i in range(num_elements):
            sub_indices = self.__unflatten_index(i, new_strides)
            orig_indices_list = []
            for j in range(len(start_indices)):
                orig_indices_list.append(start_indices[j] + sub_indices[j])
            original_indices = tuple(orig_indices_list)
            sub_data.append(self.getValue(original_indices))
        return Tensor(sub_data, new_shape)

    cpdef object format_tensor(self, list data, tuple shape):
        if len(shape) == 1:
            return data
        stride = self.__compute_num_elements(shape[1:])
        return [self.format_tensor(data[i * stride:(i + 1) * stride], shape[1:]) for i in range(shape[0])]

    def __repr__(self):
        """
        Returns a string representation of the tensor.

        :return: String representing the tensor.
        """
        formatted_data = self.format_tensor(self.__data, self.__shape)
        return f"Tensor(shape={self.__shape}, data={formatted_data})"
