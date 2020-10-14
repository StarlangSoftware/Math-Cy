from Math.Vector cimport Vector


cdef class Eigenvector(Vector):

    cdef double eigenvalue

    cpdef float getEigenvalue(self)
