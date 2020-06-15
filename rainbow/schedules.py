"""
Module containing necessary schedules
"""


class Scheduler():
    """ Scheduler base class """
    @property
    def value(self):
        """ Returns currendt variable var """
        return self._value


class LinearScheduler(Scheduler):
    """
    Class to perform a linear schedule of a variable
    """
    def __init__(self, start, end, period=None, step_size=None):
        assert not(period is None and step_size is None), \
            'Either period or step size should be provided'
        self._start = start
        self._end = end
        self._step_size = step_size
        if self._step_size is None:
            self._period = period
            self._step_size = (self._end - self._start)/self._period
        self.step = self.get_step_func()
        self._step_count = 0
        self._value = self._start

    def get_step_func(self):
        """
        Return function to increment/decrement variable by one time step
        """
        def step_pos():
            if self._value < self._end:
                self._step_count += 1
                self._value = min(self._value + self._step_size, self._end)

        def step_neg():
            if self._value > self._end:
                self._step_count += 1
                self._value = max(self._value + self._step_size, self._end)

        if self._step_size > 0:
            return step_pos
        else:
            return step_neg


class ExponentialScheduler(Scheduler):
    """
    Scheduler for epsilon during the learning process
    """
    def __init__(self, start, end, period=None, base=None):
        assert not(period is None and base is None), \
            'Either period or base should be provided'
        assert base is None or base > 0, 'Base should be greater than zero'
        self._start = start
        self._end = end
        self._base = base
        if self._base is None:
            self._period = period
            self._base = (self._end/self._start)**(1/self._period)
        self.step = self.get_step_func()
        self._step_count = 0
        self._value = self._start

    def get_step_func(self):
        """ Return function to decay/grow variable by one time step """
        def grow():
            if self._value < self._end:
                self._step_count += 1
                self._value = min(self._value*self._base, self._end)

        def decay():
            if self._value > self._end:
                self._step_count += 1
                self._value = max(self._value*self._base, self._end)

        if self._base > 1:
            return grow
        else:
            return decay
