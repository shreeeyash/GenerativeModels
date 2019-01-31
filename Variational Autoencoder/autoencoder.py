from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import numpy as np

(x_train, _),(x_test, _) = mnist.load_data()
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_test.shape, x_train.shape)

h_dim = 32

x = Input((784,))
h_x = Dense(h_dim, activation = 'relu')(x)  # sparse regularisation
g_x = Dense(784, activation = 'sigmoid')(h_x)

autoencoder = Model(x, g_x)
encoder = Model(x, h_x)

h_input = Input((h_dim,))
g_hi = autoencoder.layers[-1](h_input)
decoder = Model(h_input, g_hi)

autoencoder.compile('adam','binary_crossentropy')
autoencoder.fit(x_train,x_train,epochs = 50, batch_size = 256, shuffle = True, validation_data = (x_test,x_test))

encoder_imgs = encoder.predict(x_test)
decoder_imgs = decoder.predict(encoder_imgs)

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoder_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
