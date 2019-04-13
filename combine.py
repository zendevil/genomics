import numpy as np

x_train = np.load('GTEx_X_float64.npy')
y_train = np.load('GTEx_Y_0-4760_float64.npy')


combined = np.empty((2921,5703), float)

combined = []
for i in range(0, 2921):
    row = np.concatenate((x_train[i], y_train[i]))
    combined.append(row.tolist())

np.save('combined.npy', np.asarray(combined))
print('Done')
