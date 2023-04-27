import numpy as np
import random


x = np.arange(-10.0, 10.0, 0.4)
y = 3*np.sin(x/2) - 1
print(len(x))
for i in range(len(x)):
    print(round(x[i] + random.random(), 4), round(y[i] + random.random(), 4))

print(4)
