import pyvw
from cpyvw cimport SearchTask

cdef class L2SSum(SearchTask):
    cdef object make_examples(L2SSum self, object oscores, object inputs, 
            object cols, object Xsum_tf, object Xinp_tf)
