import pylibvw


cdef class SearchTask(object):
    cdef object vw
    cdef object sch
    cdef object blank_line
    cdef object bogus_example
    cdef object _output
    cdef object _run(SearchTask self, object your_own_input_example)
