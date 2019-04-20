
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

y = np.load('./data/debugging/3_3.npy')

labels = ['0','0.2','0.4','0.6','0.8','1.0','1.2', '2.0', '3.0', '10.0', '20.0', '30.0']
for y_arr, label in zip(y, labels):
    plt.plot(y_arr, label=label)
    

plt.ylabel('MSE')
plt.legend()
plt.savefig('data/data.png')
