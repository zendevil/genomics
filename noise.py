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
from keras.callbacks import TensorBoard
x_train = np.load('train_l1000.npy')
x_test = np.load('test_l1000.npy')

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
    
for i in range(3,4):
    n_encoder_layers = i
    n_decoder_layers = i
    autoencoder = autoencoder_num_layers(n_encoder_layers, n_decoder_layers)

    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    MSE = [] # mean square error

    for r in range(0,11): # noise factor
        noise_factor = r * 0.1
        x_train_noisy = x_train + noise_factor *\
        np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + noise_factor *\
            np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
        print('bw noisy and train',((x_train - x_train_noisy)**2).mean(axis=None) )
        autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=256,\
                        shuffle=True, validation_data=(x_test_noisy, x_test),
                        callbacks=[TensorBoard(log_dir='./tmp/tb', histogram_freq=0, write_graph=True)])

        decoded_profile = autoencoder.predict(x_train)

        mse = ((x_train - decoded_profile)**2).mean(axis=None)
        MSE.append(mse)        
            
    np.save('./data/debugged_noise/'+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'.npy', MSE)
    write_g_tex_file_1D(MSE, './data/debugged_noise/'+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'_g.tex', 'a')
    write_t_tex_file_1D(MSE, './data/debugged_noise/'+str(n_encoder_layers)+'_'+str(n_decoder_layers)+'_t.tex', 'a')
