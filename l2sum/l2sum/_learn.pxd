import pyvw
from cpyvw cimport SearchTask
import numpy as np
cimport numpy as np

INT_DTYPE = np.int
DBL_DTYPE = np.double
ctypedef np.int_t INT_DTYPE_t
ctypedef np.double_t DBL_DTYPE_t

cdef class L2SSum(SearchTask):
    cdef void update_examples(L2SSum self, object examples, int sim_start,
        np.ndarray[DBL_DTYPE_t, ndim=2] Kinp_tf, 
        object summary_i, object index,
        np.ndarray[DBL_DTYPE_t, ndim=2] Xinp_sf) 
            
