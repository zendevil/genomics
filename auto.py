# Stuff I did
# Added shuffling
# 
# TODO
# Why is loss increasing as noise in input data increases?
# Add more noise.

# This code is for analyzing 1-1 to 4-4 layer autoencoder performance for a given noise level
# across 10% to 100% samples in increments of 10% 

import sys
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
import random
from latex import *

X_GTEx = np.load('GTEx_X_float64.npy')
x_train = X_GTEx
np.random.shuffle(x_train)
def add_encoder(model, n):
    encoding_dim = 2**(5+n-1) 
    model.add(Dense(encoding_dim, activation='relu', input_shape=(943,)))
    for l in range(n-1):
        encoding_dim /= 2
        model.add(Dense(int(encoding_dim), activation='relu'))
    return model

def add_decoder(model, n):
    decoding_dim = 2**6    
    for l in range(n-1):
        model.add(Dense(int(decoding_dim), activation='relu'))
        decoding_dim *= 2
    model.add(Dense(943, activation='sigmoid'))
    return model

def print_model_layers(model):
    for layer in model.layers:
        print(layer.output_shape)
        
def autoencoder_num_layers(n_encoder_layers, n_decoder_layers):
    # 2921*943= 2754503
    model = Sequential()
    model = add_encoder(model, n_encoder_layers)
    print(type(model))
    model = add_decoder(model, n_decoder_layers)
    print(type(model))
    print_model_layers(model)
    return model

def calc_mse_all_sample_sizes(autoencoder, x_train, noise_factor):
    x_noisy = x_train + noise_factor *\
            np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    np.random.shuffle(x_noisy)
    mse_across_sample_sizes = []
    for s in range(1,11): # samples    
        samples_ratio = s * 0.1
        samples = int((samples_ratio)*len(X_GTEx))

        x_train = x_train[:samples]
        
        x_noisy = x_noisy[:samples]
        
        x_test = x_train

        autoencoder.fit(x_noisy, x_train, epochs=10, batch_size=256,\
                        shuffle=True, validation_data=(x_test, x_test))

        decoded_profile = autoencoder.predict(x_train)

        mse = ((x_train - decoded_profile)**2).mean(axis=None) / samples
        mse_across_sample_sizes.append(mse)
        print('noise =', noise_factor, 'samples =', samples,\
                  'normalized mse =', mse)
    print('mse sample sizes', len(mse_across_sample_sizes))
    return mse_across_sample_sizes

for i in range(3,4):
    n_encoder_layers = i
    n_decoder_layers = i
    autoencoder = autoencoder_num_layers(n_encoder_layers, n_decoder_layers)

    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    MSE = [] # mean square error

    # for r in range(10,11): # noise factor
    #     noise_factor = r * 0.1
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 0  ))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 0.2))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 0.4))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 0.6))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 0.8))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 1.0))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 1.2))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 2.0))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 3.0))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 10.0))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 20.0))
    MSE.append(calc_mse_all_sample_sizes(autoencoder, x_train, 30.0))
    
    dir_ = './data/noise/'
    np.save(dir_+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'.npy', MSE)
    # write_g_tex_file(MSE, './data/noise/'+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'_g.tex', 'a')
    # write_t_tex_file(MSE, './data/noise/'+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'_t.tex', 'a')
    print('loss data saved in', dir_) 
