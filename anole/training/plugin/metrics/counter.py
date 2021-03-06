from .base_metrics import BaseMetric

__all__ = ['Mean', 'Sum', 'Min', 'Max']


class Mean(BaseMetric):
    """
    The standalone mean metric.
    This utility metric is a general purpose metric that can be used to keep
    track of the mean of a sequence of values.
    """

    def __init__(self):
        """
        Creates an instance of the mean metric.
        This metric in its initial state will return a mean value of 0.
        The metric can be updated by using the `update` method while the mean
        can be retrieved using the `result` method.
        """
        super().__init__()
        self.summed: float = 0.0
        self.weight: float = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        """
        Update the running mean given the value.
        The value can be weighted with a custom value, defined by the `weight`
        parameter.
        :param value: The value to be used to update the mean.
        :param weight: The weight of the value. Defaults to 1.
        :return: None.
        """
        value = float(value)
        weight = float(weight)
        self.summed += value * weight
        self.weight += weight

    def result(self) -> float:
        """
        Retrieves the mean.
        Calling this method will not change the internal state of the metric.
        :return: The mean, as a float.
        """
        if self.weight == 0.0:
            return 0.0
        return self.summed / self.weight

    def reset(self) -> None:
        """
        Resets the metric.
        :return: None.
        """
        self.summed = 0.0
        self.weight = 0.0

    def __add__(self, other: 'Mean') -> "Mean":
        """
        Return a metric representing the weighted mean of the 2 means.
        :param other: the other mean
        :return: The weighted mean"""
        res = Mean()
        res.summed = self.summed + other.summed
        res.weight = self.weight + other.weight
        return res


class Sum(BaseMetric):
    """
    The standalone sum metric.
    This utility metric is a general purpose metric that can be used to keep
    track of the sum of a sequence of values.
    Beware that this metric only supports summing numbers and the result is
    always a float value, even when `update` is called by passing `int`s only.
    """

    def __init__(self):
        """
        Creates an instance of the sum metric.
        This metric in its initial state will return a sum value of 0.
        The metric can be updated by using the `update` method while the sum
        can be retrieved using the `result` method.
        """
        super().__init__()
        self.summed: float = 0.0

    def update(self, value: float) -> None:
        """
        Update the running sum given the value.
        :param value: The value to be used to update the sum.
        :return: None.
        """
        self.summed += float(value)

    def result(self) -> float:
        """
        Retrieves the sum.
        Calling this method will not change the internal state of the metric.
        :return: The sum, as a float.
        """
        return self.summed

    def reset(self) -> None:
        """
        Resets the metric.
        :return: None.
        """
        self.summed = 0.0


class Min(BaseMetric):
    """
    The standalone sum metric.
    This utility metric is a general purpose metric that can be used to keep
    track of the sum of a sequence of values.
    Beware that this metric only supports summing numbers and the result is
    always a float value, even when `update` is called by passing `int`s only.
    """

    def __init__(self):
        """
        Creates an instance of the sum metric.
        This metric in its initial state will return a sum value of 0.
        The metric can be updated by using the `update` method while the sum
        can be retrieved using the `result` method.
        """
        super().__init__()
        self.min_val: float = float('inf')

    def update(self, value: float) -> bool:
        """
        Update the running sum given the value.
        :param value: The value to be used to update the sum.
        :return: None.
        """
        if value < self.min_val:
            self.min_val = value
            return True
        else:
            return False

    def result(self) -> float:
        """
        Retrieves the sum.
        Calling this method will not change the internal state of the metric.
        :return: The sum, as a float.
        """
        return self.min_val

    def reset(self) -> None:
        """
        Resets the metric.
        :return: None.
        """
        self.min_val = float('inf')


class Max(BaseMetric):
    """
    The standalone sum metric.
    This utility metric is a general purpose metric that can be used to keep
    track of the sum of a sequence of values.
    Beware that this metric only supports summing numbers and the result is
    always a float value, even when `update` is called by passing `int`s only.
    """

    def __init__(self):
        """
        Creates an instance of the sum metric.
        This metric in its initial state will return a sum value of 0.
        The metric can be updated by using the `update` method while the sum
        can be retrieved using the `result` method.
        """
        super().__init__()
        self.max_val: float = -float('inf')

    def update(self, value: float) -> bool:
        """
        Update the running sum given the value.
        :param value: The value to be used to update the sum.
        :return: None.
        """
        if value > self.max_val:
            self.max_val = value
            return True
        else:
            return False

    def result(self) -> float:
        """
        Retrieves the sum.
        Calling this method will not change the internal state of the metric.
        :return: The sum, as a float.
        """
        return self.max_val

    def reset(self) -> None:
        """
        Resets the metric.
        :return: None.
        """
        self.max_val = -float('inf')
