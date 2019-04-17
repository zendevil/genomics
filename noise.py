#TODO
# shuffle data when training
# make test data
import sys
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
import random
from latex import *

X_GTEx = np.load('GTEx_X_float64.npy')
x_train = X_GTEx

def add_encoder(model, n):
    encoding_dim = 2**(5+n-1) # 
    model.add(Dense(encoding_dim, activation='relu', input_shape=(943,)))
    for l in range(n-1):
        encoding_dim /= 2
        model.add(Dense(int(encoding_dim), activation='relu'))
    return model

def add_decoder(model, n):
    decoding_dim = 2**6    
    for l in range(n-1):
        model.add(Dense(int(decoding_dim), activation='sigmoid'))
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

def calc_mse_all_sample_sizes(x_train, noise_factor):
    x_noisy = x_train + noise_factor *\
            np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    mse_across_sample_sizes = []
    for s in range(1,11): # samples    
        samples_ratio = s * 0.1
        samples = int((samples_ratio)*len(X_GTEx))

        x_train = x_train[:samples]
        x_noisy = x_noisy[:samples]
        x_test = x_train

        autoencoder.fit(x_noisy, x_train, epochs=50, batch_size=256,\
                        shuffle=True, validation_data=(x_test, x_test))

        decoded_profile = autoencoder.predict(x_train)

        mse = ((x_train - decoded_profile)**2).mean(axis=None) / samples
        mse_across_sample_sizes.append(mse)
        print('noise =', noise_factor, 'samples =', samples,\
                  'normalized mse =', mse)
    print('mse sample sizes', len(mse_across_sample_sizes))
    return mse_across_sample_sizes

def calc_mse(x_train, noise_factor):
    x_noisy = x_train + noise_factor *\
            np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    autoencoder.fit(x_noisy, x_train, epochs=50, batch_size=256,\
                        shuffle=True, validation_data=(x_train, x_train))

    decoded_profile = autoencoder.predict(x_train)

    mse = ((x_train - decoded_profile)**2).mean(axis=None)
    return mse
    
for i in range(1,5):
    n_encoder_layers = i
    n_decoder_layers = i
    autoencoder = autoencoder_num_layers(n_encoder_layers, n_decoder_layers)

    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    MSE = [] # mean square error

    for r in range(0,11): # noise factor
        noise_factor = r * 0.1
        MSE.append(calc_mse(x_train, noise_factor))        
            
    np.save('./data/noise/'+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'.npy', MSE)
    write_g_tex_file_1D(MSE, './data/noise/'+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'_g.tex', 'a')
    write_t_tex_file_1D(MSE, './data/noise/'+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'_t.tex', 'a')
