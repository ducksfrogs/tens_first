import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.arange(0,6,0.1)
y = np.sin(x)


plt.plot(x,y)


y2 = np.cos(x)


plt.plot(x,y, label='sin')
plt.plot(x, y2,linestyle='--', label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sin & Cos')
plt.legend()




