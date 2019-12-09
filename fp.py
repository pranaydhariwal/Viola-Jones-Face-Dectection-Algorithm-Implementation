from numpy import *
import math
import matplotlib.pyplot as plt

fp = [0.111444, 0.099000, 0.04580, 0.05490]
fn = [0.622881, 0.654309, 0.63345, 0.61911]
rounds = [1, 3, 5, 10]

plt.plot(rounds, fp, 'r', label="False Positive") # plotting t, a separately
plt.plot(rounds, fn, 'b', label="False Negative") # plotting t, b separately
plt.legend(loc="best")
plt.xlabel("Rounds")
plt.ylabel("Rates")
plt.show()

