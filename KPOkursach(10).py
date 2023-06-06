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

class Modelling(ABC):
  def __init__(self, gen: RandomNumberGenerator, estimations: list, N:int, M:int, truth_value:float):
    self.gen = gen
    self.estimations = estimations
    self.N = N
    self.M = M
    self.truth_value = truth_value

    # Здесь будут храниться выборки оценок
    self.estimations_sample = np.zeros((self.M, len(self.estimations)), dtype=np.float64)

  # Метод, оценивающий квадрат смещения оценок
  def estimate_bias_sqr(self):
    return np.array([(Mean().estimate(self.estimations_sample[:,i]) - self.truth_value) ** 2 for i in range(len(self.estimations))])

  # Метод, оценивающий дисперсию оценок
  def estimate_var(self):
    return np.array([Var().estimate(self.estimations_sample[:,i]) for i in range(len(self.estimations))])

  # Метод, оценивающий СКО оценок
  def estimate_mse(self):
    return self.estimate_bias_sqr() + self.estimate_var()

  def get_samples(self):
    return self.estimations_sample

  def get_sample(self):
    return self.gen.get(self.N)

  def run(self):
    for i in range(self.M):
      sample = self.get_sample()
      self.estimations_sample[i, :] = [e.estimate(sample) for e in self.estimations]

class SmoothedRandomVariable():
    def _k(self, x):
        z = x
        return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi))

    def _K(self, x):
        z = x
        if z <= 0:
            return 0.852 * math.exp(-math.pow((-z + 1.5774) / 2.0637, 2.34))
        return 1 - 0.852 * math.exp(-math.pow((z + 1.5774) / 2.0637, 2.34))

    def h(self, x):
        return -x * (math.exp(-0.5 * x * x) / (math.sqrt(2 * math.pi)))

    def get_h(self):
        hn = np.std(self.sample, ddof=1) / (len(self.sample) - 1)
        delta = 1.0
        while delta >= 0.001:
            s = 0.0
            for i in range(len(self.sample)):
                num = 0.0
                div = 0.0
                for j in range(len(self.sample)):
                    if i != j:
                        diff = (self.sample[j] - self.sample[i]) / hn
                        num += self.h(diff) * (self.sample[j] - self.sample[i])
                        div += self._k(diff)
                if div == 0.0:
                    continue
                s += num / div
            new_hn = - (1 / len(self.sample)) * s
            delta = abs(new_hn - hn)
            hn = new_hn
        return hn

    def __init__(self, sample):
        self.sample = sample
        self.h = self.get_h()

    def pdf(self, x):
        return np.mean([self._k((x - y) / self.h) for y in self.sample]) / self.h

    def cdf(self, x):
        return np.mean([self._K((x - y) / self.h) for y in self.sample])

#объем выборки
n = 1000
#кол-во ревыборок
m = 100
#параметр сдвига
location = 0

#создаем распределение с параметрами
main_rv = CauchyRandomVariable(location,1)
#создаем генератор, чтобы получить выборку с распределением выше
gen = SimpleRandomNumberGenerator(main_rv)
#получаем выборку
sample1 = gen.get(n)
#передаем выборку в бутстреп
main_rv1 = NonParametricRandomVariable(sample1)

#тут указываешь второе распределение и его параметры
outlier_rv = UniformRandomVariable(-10,10)
#Симметричные выбросы:
#outlier_rv = CauchyRandomVariable(0,5)
gen1 = SimpleRandomNumberGenerator(outlier_rv)
sample2 = gen1.get(n)
outlier_rv1 = NonParametricRandomVariable(sample2)

#генератор с выбросами
outlier_gen = TukeyRandomNumberGenerator(main_rv1, outlier_rv1)

rv = CauchyRandomVariable()
generator = SimpleRandomNumberGenerator(rv)
sample3 = generator.get(n)
rv1 = NonParametricRandomVariable(sample3)

#генератор без выбросов
generator1 = SimpleRandomNumberGenerator(rv1)

#меняем генератор (с generator1 на outlier_gen)
modelling = Modelling(generator1, [Mean(), Median()],n, m, location)
modelling.run()
mses = modelling.estimate_mse()
#СКО оценок
print(mses)
#вторая оценка хуже
print("Первая оценка лучше в ",mses[1] / mses[0])
#первая оценка хуже
print("Вторая оценка лучше в ",mses[0] / mses[1])

samples = modelling.get_samples()
POINTS = 100
#синий - график для средней
#оранжевый - график для медианы
for i in range(samples.shape[1]):
  sample = samples[:, i]
  X_min = min(sample)
  X_max = max(sample)
  x = np.linspace(X_min, X_max, POINTS)
  srv = SmoothedRandomVariable(sample)
  y = np.vectorize(srv.pdf)(x)
  plt.plot(x, y)
plt.show()