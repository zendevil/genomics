#TODO

# shuffle data when training
# make test data

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

encoded = Dense(128, activation='relu', #activity_regularizer=regularizers.l1(10e-5)
)(input_profile)
encoded = Dense(64, activation='relu', #activity_regularizer=regularizers.l1(10e-5)
)(encoded)
encoded = Dense(32, activation='relu', # activity_regularizer=regularizers.l1(10e-5)
)(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(64, activation='sigmoid')(encoded)
decoded = Dense(128, activation='sigmoid')(decoded)
decoded = Dense(943, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_profile, decoded)

# this model maps an input to its encoded representation
# encoder = Model(input_profile, encoded)

# # create a placeholder for an encoded (32-dimensional) input
# encoded_input = Input(shape=(encoding_dim,))
# # retrieve the last layer of the autoencoder model

# decoder_layer = autoencoder.layers[-1]

# # create the decoder model
# decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

max_ =  14.069335820245042
min_ = -8.195025678439624

def pick_random_indices(array, ratio):
    s = set()
    for i in range(len(array)):
        s.add(i)
    for i in range(int(len(array)*(1-ratio))):
        s.remove(random.sample(s, 1)[0])
    return s
    
        
def add_noise(profile, ratio):
    print(len(profile))
    for x in range(len(profile)):
        sample = profile[x]
        noise_indices = pick_random_indices(sample, ratio)
        for i in range(len(noise_indices)):
            index = noise_indices.pop()
            sample[index] = random.uniform(min_, max_)
        profile[x] = sample

    return profile

import numpy as np

X_GTEx = np.load('GTEx_X_float64.npy')
Y_GTEx = np.load('GTEx_Y_0-4760_float64.npy')
x_test = Y_GTEx

def write_g_prefix(g, data, noise_ratio):
    max_ = max(data)
    min_ = min(data)

    diff = max_ - min_
    g.write('\\begin{tikzpicture}\n')
    g.write('\\begin{axis}[\n')
    g.write('title={Sample size of training data plotted against accuracy at '+str(noise_ratio)+' noise ratio},\n')
    g.write('xlabel={Number of Samples},\n')
    g.write('ylabel={Mean Square Error between all original and denoised samples},\n')
    g.write('xmin=0, xmax=3000,\n')
    g.write('ymin='+str(min_)+', ymax='+str(max_)+',\n')
    g.write('xtick={0,500,1000,1500,2000,2500,3000},\n')
    g.write('ytick={'+str(min_)+','+str(min_+1*diff)+','+str(min_+2*diff)+','+str(min_+3*diff)+','+str(min_+4*diff)+','+str(min_+5*diff)+','+str(min_+6*diff)+','+str(min_+7*diff)+','+str(min_+8*diff)+','+str(min_+9*diff)+','+str(min_+10*diff)+'},\n')
    g.write('legend pos=north west,\n')
    g.write('ymajorgrids=true,\n')
    g.write('grid style=dashed,\n')
    g.write(']\n\n')
    g.write('\\addplot[\n')
    g.write('color=blue,\n')
    g.write('mark=square,\n')
    g.write(']\n')
    g.write('coordinates {\n\n')

def write_t_prefix(t):
    t.write('\\npdecimalsign{.}\n')
    t.write('\\nprounddigits{2}\n')
    t.write('\\begin{tabu} to 0.8\\textwidth { | X[l] | X[r] |}\n')
    t.write('\\hline\n')
    t.write('samples &  MSE\\\\\n')
    t.write('\\hline\n')

def write_g_suffix(g):
    g.write('    };\n')
    g.write('\\end{axis}\n')
    g.write('\\end{tikzpicture}')

def write_t_suffix(t):
    t.write('\\end{tabu}\n')
    t.write('\\npnoround\n')

MSE = []

for r in range(0,11): # noise ratio
    noise_ratio = r * 0.1
    x_train = X_GTEx
    x_noisy = add_noise(x_train, noise_ratio)
    MSE.append([])
    for s in range(1,11): # samples
        samples_ratio = s * 0.1
        samples = int(samples_ratio*len(X_GTEx))
        x_train = x_train[:samples]
        x_test = x_train
        x_noisy = x_noisy[:samples]
        print('x_train shape', x_train.shape)
        print('x_noisy shape', x_noisy.shape)
        autoencoder.fit(x_noisy, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
        # note that we take them from the train set
        # encoded_profile = encoder.predict(x_train)
        decoded_profile = autoencoder.predict(x_train)
        mse = ((x_train - decoded_profile)**2).mean(axis=None)/samples
        MSE[r].append(mse)
        print('noise =',noise_ratio, 'samples =',samples,'normalized mse =',mse)
print(MSE)


g = open('./uniform_noise/triple/graphs.tex', 'a')
g.write('\\documentclass{article}\n\
\\usepackage{tikz}\n\
\\usepackage{pgfplots}\n\
\\usepackage{textcomp}\n\
\\usepackage{array}\n\
\\usepackage{tabu}\n\
\\usepackage{numprint}\n\
\\begin{document}')
t = open('./uniform_noise/triple/'+'tables.tex', 'a')
t.write('\\documentclass{article}\n\
\\usepackage{tikz}\n\
\\usepackage{pgfplots}\n\
\\usepackage{textcomp}\n\
\\usepackage{array}\n\
\\usepackage{tabu}\n\
\\usepackage{numprint}\n\
\\begin{document}')
for r in range(0,11):
    noise_ratio = r * 0.1
    
    write_g_prefix(g, MSE[r], noise_ratio) # passing in M[r] to 
    write_t_prefix(t) # calculate ticks. 
    for s in range(1,11):
        samples_ratio = s * 0.1
        samples = int((samples_ratio)*len(X_GTEx))
        g.write('('+str(samples)+', '+str(MSE[r][s-1])+')\n')
        t.write(str(samples)+' & '+str(MSE[r][s-1]) +'\\'+'\\' + '\n'+'\hline\n')
    write_g_suffix(g)
    write_t_suffix(t)


g.write('\n\
\\end{document}\n')
t.write('\\end{document}')
    
g.close()
t.close()

# print('max', np.amax(encoded_profile))
# print('min', np.amin(encoded_profile))

# # use Matplotlib (don't ask)
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# n = 10  # how many profiles we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(127, 23))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_profile[i].reshape(127, 23))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.savefig('./comparison50.png')
