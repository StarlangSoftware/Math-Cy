cdef class Vector(object):

    cdef int __size
    cdef list __values

    cpdef initAllSame(self, int size, double x)
    cpdef initAllZerosExceptOne(self, int size, int index, double x)
    cpdef Vector biased(self)
    cpdef add(self, double x)
    cpdef insert(self, int pos, double x)
    cpdef remove(self, int pos)
    cpdef clear(self)
    cpdef double sumOfElements(self)
    cpdef int maxIndex(self)
    cpdef sigmoid(self)
    cpdef Vector skipVector(self, int mod, int value)
    cpdef addVector(self, Vector v)
    cpdef subtract(self, Vector v)
    cpdef Vector difference(self, Vector v)
    cpdef double dotProduct(self, Vector v)
    cpdef double dotProductWithSelf(self)
    cpdef Vector elementProduct(self, Vector v)
    cpdef divide(self, double value)
    cpdef multiply(self, double value)
    cpdef Vector product(self, double value)
    cpdef l1Normalize(self)
    cpdef double l2Norm(self)
    cpdef double cosineSimilarity(self, Vector v)
    cpdef int size(self)
    cpdef double getValue(self, int index)
    cpdef setValue(self, int index, double value)
    cpdef addValue(self, int index, double value)