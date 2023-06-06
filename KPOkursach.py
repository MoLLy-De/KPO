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


