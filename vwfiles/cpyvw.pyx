import pylibvw

cdef class SearchTask(object):
    def __cinit__(self, vw, sch, num_actions):
        self.vw = vw
        self.sch = sch
        self.blank_line = self.vw.example("")
        self.blank_line.finish()
        self.bogus_example = self.vw.example("1 | x")

    def __del__(self):
        self.bogus_example.finish()
        pass

    cdef object _run(SearchTask self, object your_own_input_example):
        pass

    def _call_vw(self, my_example, isTest, useOracle=False): # run_fn, setup_fn, takedown_fn, isTest):
        self._output = None
        self.bogus_example.set_test_only(isTest)
        def run(): self._output = self._run(my_example)
        setup = None
        takedown = None
        if callable(getattr(self, "_setup", None)): setup = lambda: self._setup(my_example)
        if callable(getattr(self, "_takedown", None)): takedown = lambda: self._takedown(my_example)
        self.sch.set_structured_predict_hook(run, setup, takedown)
        self.sch.set_force_oracle(useOracle)
        self.vw.learn(self.bogus_example)
        self.vw.learn(self.blank_line) # this will cause our ._run hook to get called
        
    def learn(self, data_iterator):
        for my_example in data_iterator.__iter__():
            self._call_vw(my_example, isTest=False);

    def example(self, initStringOrDict=None, labelType=pylibvw.vw.lDefault):
        """TODO"""
        if self.sch.predict_needs_example():
            return self.vw.example(initStringOrDict, labelType)
        else:
            return self.vw.example(None, labelType)
            
    def predict(self, my_example, useOracle=False):
        self._call_vw(my_example, isTest=True, useOracle=useOracle);
        return self._output


