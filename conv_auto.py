from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras import backend as K
import numpy as np

input_profile = Input(shape=(24, 24, 1))
#x = ZeroPadding2D(padding=((0,1), (0,1)))(input_profile)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_profile)
print('Conv2d', x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print('MaxPooling2D1', x.shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print('Conv2D2', x.shape)
x = MaxPooling2D((2, 2), padding='same')(x)
print('MaxPooling2D2', x.shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print('Conv2D3', x.shape)
encoded = MaxPooling2D((2, 2), padding='same')(x)
print('MaxPooling2D3', x.shape)
# at this point the representation is (4, 42754503, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
print(x.shape)
x = UpSampling2D((2, 2))(x)
print(x.shape)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
print(x.shape)
x = UpSampling2D((2, 2))(x)
print(x.shape)
x = ZeroPadding2D(padding=(1,1))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
print(x.shape)
x = UpSampling2D((2, 2))(x)
print(x.shape)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
print(decoded.shape)
autoencoder = Model(input_profile, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

X_GTEx = np.load('GTEx_X_float64.npy')
x_train = X_GTEx 
x_train = np.reshape(np.pad(x_train.flatten(), (0,505), mode='constant', constant_values=0), (4783, 24, 24, 1))
from keras.callbacks import TensorBoard

autoencoder.fit(x_train, x_train,\
                epochs=50, batch_size=127,\
                shuffle=True, validation_data=(x_train, x_train),\
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

decoded_profile = autoencoder.predict(x_train)
mse = ((x_train - decoded_profile)**2).mean(axis=None)
print(mse)
f = open('./conv.txt', 'w')
f.write(str(mse))
