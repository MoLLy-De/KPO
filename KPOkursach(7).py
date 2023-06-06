import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
import math
import random
import statistics

class RandomVariable(ABC):
    @abstractmethod
    def pdf(self, x):
        pass

    @abstractmethod
    def cdf(self, x):
        pass

    @abstractmethod
    def quantile(self, alpha):
        pass

class CauchyRandomVariable(RandomVariable):
    def __init__(self, loc=0, scale=1):
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return 1 / (math.pi * self.scale * (1 + ((x - self.loc) / self.scale) ** 2))

    def cdf(self, x):
        return 0.5 + math.atan((x - self.loc) / self.scale) / math.pi

    def quantile(self, alpha):
        return self.loc + self.scale * math.tan(math.pi * (alpha - 0.5))

class UniformRandomVariable(RandomVariable):
  def __init__(self, a=0, b=1) -> None:
    super().__init__()
    self.a = a
    self.b = b

  def pdf(self, x):
    if x >= self.a and x <= self.b:
      return 1 / (self.b - self.a)
    else:
      return 0

  def cdf(self, x):
    if x <= self.a:
      return 0
    elif x >=self.b:
      return 1
    else:
      return (x - self.a) / (self.b - self.a)

  def quantile(self, alpha):
    return self.a + alpha * (self.b - self.a)

class NonParametricRandomVariable(RandomVariable):
    def __init__(self, source_sample) -> None:
        super().__init__()
        self.source_sample = sorted(source_sample)

    def pdf(self, x):
        if x in self.source_sample:
            return float('inf')
        return 0

    @staticmethod
    def heaviside_function(x):
        if x > 0:
            return 1
        else:
            return 0

    def cdf(self, x):
        return np.mean(np.vectorize(self.heaviside_function)(x - self.source_sample))

    def quantile(self, alpha):
        index = int(alpha * len(self.source_sample))
        return self.source_sample[index]

class RandomNumberGenerator(ABC):
    def __init__(self, random_variable: RandomVariable):
        self.random_variable = random_variable

    @abstractmethod
    def get(self, N):
        pass

class SimpleRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, random_variable: RandomVariable):
        super().__init__(random_variable)

    def get(self, N):
        us = np.random.uniform(0, 1, N)
        return np.vectorize(self.random_variable.quantile)(us)

class TukeyRandomNumberGenerator(RandomNumberGenerator):
    def __init__(self, main_random_variable: RandomVariable, outlier_random_variable: RandomVariable, e:float = 0.1):
        super().__init__(main_random_variable)
        self.outlier_random_variable = outlier_random_variable
        self.e = e

    def get(self, N):
        us = np.random.uniform(0, 1, N)
        sample = []
        for x in us:
            if x < self.e:
                sample.append(self.outlier_random_variable.quantile(random.random()))
            else:
                sample.append(self.random_variable.quantile(random.random()))
        return sample

class Estimation(ABC):
  @abstractmethod
  def estimate(self, sample):
    pass

class Mean(Estimation):
  def estimate(self, sample):
    return statistics.mean(sample)

class Median(Estimation):
  def estimate(self, sample):
    return statistics.median(sample)

class Var(Estimation):
  def estimate(self, sample):
    return statistics.variance(sample)