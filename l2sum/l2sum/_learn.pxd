import pyvw
from cpyvw cimport SearchTask
import numpy as np
cimport numpy as np

INT_DTYPE = np.int
DBL_DTYPE = np.double
ctypedef np.int_t INT_DTYPE_t
ctypedef np.double_t DBL_DTYPE_t

cdef class L2SSum(SearchTask):
    cdef int score_func
    cdef double alpha
    cdef void update_examples(L2SSum self, object examples, int sim_start,
        int int_start, int use_interactions,
        np.ndarray[DBL_DTYPE_t, ndim=2] Kinp_tf, 
        object summary_i, object index,
        np.ndarray[DBL_DTYPE_t, ndim=2] Xinp_sf) 
    cdef double _compute_r(L2SSum self, list model_ngrams, 
            object summary_tokens)
    cdef double _compute_p(L2SSum self, list model_ngrams, 
            object summary_tokens)
    cdef double _compute_f1(L2SSum self, list model_ngrams, 
            object summary_tokens)
            
