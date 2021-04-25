cdef class DiscreteDistribution(dict):

    cdef double __sum

    cpdef addItem(self, str item)
    cpdef removeItem(self, str item)
    cpdef addDistribution(self, DiscreteDistribution distribution)
    cpdef removeDistribution(self, DiscreteDistribution distribution)
    cpdef double getSum(self)
    cpdef int getIndex(self, str item)
    cpdef bint containsItem(self, str item)
    cpdef str getItem(self, int index)
    cpdef int getValue(self, int index)
    cpdef int getCount(self, str item)
    cpdef str getMaxItem(self)
    cpdef str getMaxItemIncludeTheseOnly(self, list includeTheseOnly)
    cpdef double getProbability(self, str item)
    cpdef dict getProbabilityDistribution(self)
    cpdef double getProbabilityLaplaceSmoothing(self, str item)
    cpdef double entropy(self)
