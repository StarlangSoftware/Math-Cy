cdef class Tensor:
    cdef double[:] __data
    cdef tuple __shape
    cdef tuple __strides

    cpdef float getValue(self, tuple indices)
    cpdef void setValue(self, tuple indices, float value)
    cpdef Tensor broadcast_to(self, tuple target_shape)
    cpdef Tensor add(self, Tensor other)
    cpdef Tensor subtract(self, Tensor other)
    cpdef Tensor hadamardProduct(self, Tensor other)
    cpdef Tensor multiply(self, Tensor other)
    cpdef Tensor partial(self, tuple start_indices, tuple end_indices)
    cpdef Tensor transpose(self, tuple axes=?)
    cpdef Tensor reshape(self, tuple new_shape)
    cpdef Tensor concat(self, Tensor other, int dimension)
    cpdef Tensor get(self, tuple dimensions)
    cpdef object format_tensor(self, list data, tuple shape)

    cdef list __flatten(self, object data)
    cdef tuple __infer_shape(self, object data)
    cdef int __compute_num_elements(self, tuple shape)
    cdef tuple __compute_strides(self, tuple shape)
    cdef tuple __unflatten_index(self, int flat_index, tuple strides)
    cdef tuple __broadcast_shape(self, tuple shape1, tuple shape2)
    cdef void __validate_indices(self, tuple indices)
    cdef list __transpose_flattened_data(self, tuple axes, tuple new_shape)
