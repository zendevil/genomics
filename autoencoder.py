#TODO
# combine x and y train for autoencoder.

import sys
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import random
# 2921*943= 2754503
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_profile = Input(shape=(943,))
# "encoded" is the encoded representation of the input

encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_profile)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(943, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_profile, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_profile, encoded)


# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

max =  14.069335820245042
min = -8.195025678439624

def pick_random_indices(array, ratio):
    s = set()
    for i in range(len(array)):
        s.add(i)
    for i in range(int(len(array)*ratio)):
        s.remove(random.sample(s, 1)[0])
    return s
    
        
def add_noise(profile):
    for x in range(943):
        sample = profile[x]
        noise_indices = pick_random_indices(sample, ratio=0.2)
        for i in range(len(noise_indices)):
            index = noise_indices.pop()
            sample[index] = random.uniform(min, max)
        profile[x] = sample

    return profile



import numpy as np


      
X_GTEx = np.load('GTEx_X_float64.npy')
Y_GTEx = np.load('GTEx_Y_0-4760_float64.npy')

x_train = X_GTEx
print(x_train.shape)
x_noisy = add_noise(x_train)
x_test = Y_GTEx

print('max', np.amax(x_train))
print('min', np.amin(x_train))

print(X_GTEx.shape)
print('Y_GTEx', Y_GTEx.shape)

# x_train = np.reshape(X_GTEx, X_GTEx.reshape(1, -1))
# x_test = np.reshape(Y_GTEx, X_GTEx.reshape(1, -1))
# print(x_train.shape)
# print(x_test.shape)


# training autoencoder
print(x_train.shape)
autoencoder.fit(x_noisy, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_train, x_train))


# note that we take them from the *test* set
encoded_profile = encoder.predict(x_train)
decoded_profile = decoder.predict(encoded_profile)



print('max', np.amax(encoded_profile))
print('min', np.amin(encoded_profile))

# use Matplotlib (don't ask)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = 10  # how many profiles we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(127, 23))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_profile[i].reshape(127, 23))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('./comparison50.png')
