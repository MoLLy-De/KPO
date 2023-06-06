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

