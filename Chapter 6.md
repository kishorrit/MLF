# Chapter 6 Generative models

Generative models generate new data. In a way, they are the exact opposite of the models we dealt with before. While an image classifier takes in a high dimensional input, the image, and outputs a low dimensional output such as the content of the image, a generative model goes exactly the other way around. It might for example draw images from the description of what is in them. So far, generative models are mostly used in image applications and they are still very experimental. Yet, there already have been several applications which caused uproar. In 2017, so called "DeepFakes" appeared on the internet. So called Generative Adversarial Models (GANs) which we will cover later in this chapter were used to generate pornographic videos featuring famous celebrities. The year before, researchers demoed a system in which they could generate videos of politicians saying anything the researcher wanted them to say, complete with realistic mouth movements and facial expressions. But there are positive applications as well. Generative models are especially useful if data is sparse. They can generate realistic data other models can train on. They can "translate" images, for example from satellite images to street maps. They can generate code from website screenshots. They can even be used to combat unfairness and discrimination in machine learning models. 

In the field of finance, data is frequently sparse. Think about the fraud case from chapter 2. There were not that many frauds in the dataset, so the model had a hard time detecting frauds. Usually, engineers would create synthetic data, by thinking about how fraud could be done. Machine learning models can do this themselves however. And in the process, they might even discover some useful features for fraud detection.

In algorithmic trading, data is frequently generated in simulators. Want to know how your algorithm would do in a global selloff? There are not that many global selloffs thankfully, so engineers at quant firms spend a lot of time creating simulations of selloffs. These simulators are often biased by the engineers experience and their feelings about what a selloff should look like. But what if models could learn what a selloff fundamentally looks like, and then create data describing an infinite amount of selloffs?

# Understanding autoencoders
https://blog.keras.io/building-autoencoders-in-keras.html

https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

http://tiao.io/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/

Technically, autoencoders are not generative models since they can not create completely new kinds of data. Yet, variational autoencoders, a minor tweak to vanilla autonecoders, can. So it makes sense to first understand autoencoders by themselves, before adding the generative element. Autoencoders by themselves also have some interesting properties which can be exploited for applications like detecting credit card fraud. 

Given an input $x$, an autoencoder learns how to output $x$. It aims to find a function $f$ so that:

$$x = f(x)$$

This might sound trivial at first, but the trick is that autoencoders have a bottleneck. The middle hidden layer size is smaller than the size of the input $x$. Therefore, the model has to learn a compressed representation that captures all important elements of $x$ in a smaller vector. 

![Autoncoder Scheme](./assets/autoencoder_scheme.png)
Caption: Autoencoder Scheme

This compressed representation aims to capture the 'essence' of the input. And that turns out to be useful. We might for example want to capture, what essentially distinguishes a fraudulent from a genuine transaction. Vanilla autoencoders accomplish something to standard principal component analysis (PCA): They allow us to reduce the dimensionality of our data and focus on what matters. But in contrast to PCA, autoencoders can be extended to generate more data of a certain type. They can better deal with image or video data since they can make use of the spatiality of data using convolutional layers. In this section, we will build two autoencoders. The first for hand written digits from the MNIST dataset. Generative models are easier to debug and understand for visual data because humans are intuitively good ad judging if two pictures show something similar, but less good at judging abstract data. We will then use the same autoencoder for a fraud detection task.

## Autoencoder for MNIST 
Lets start with a simple autoencoder for the MNIST dataset of hand drawn digits. An MNIST image is 28 by 28 pixels and can be flattened into a vector of 784 (equals 28 * 28) elements. We will compress this into a vector with only 32 elements by using an autoencoder.

You can find the code for the MNIST autoencoder and variational autoencoder under the following URL:
https://www.kaggle.com/jannesklaas/mnist-autoencoder-vae

We set the encoding dimensionality hyperparameter now so we can use it later:

```Python 
encoding_dim = 32 
```

We construct the autoecoder using Keras functional API. While a simple autencoder could be constructed using the sequential API, it is a good refresher on how the functional API works.

First, we import the `Model` class that allows us to create functional API models. We also need to import `Input` and `Dense` layers. Remember that the functional API needs a separate input layer while the sequential API does not need one.
```Python 
from keras.models import Model
from keras.layers import Input, Dense
```

Now we are chaining up the autoencoder's layers: An `Input` layer, followed by a `Dense` layer which encodes the image to a smaller representation. This is followed by a decoding `Dense` layer which aims to reconstruct the original image.
```Python 
input_img = Input(shape=(784,))

encoded = Dense(encoding_dim, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)
``` 

After we have created and chained up the layers, we create a model which maps from the input to the decoded image.

```Python 
autoencoder = Model(input_img, decoded)
``` 

To get a better idea of what is going on, we can plot a visualization of the resulting autoencoder model. As of writing this code snippet will not work in Kaggle, but future versions of the Kaggle Kernels editor might change this

```Python 
from keras.utils import plot_model
plot_model(autoencoder, to_file='model.png', show_shapes=True)
```

This is our autencoder:

![Autoencoder Model](./assets/autoencoder_model.png)

```Python 
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
```
Caption: A simple autoencoder for MNIST

To train this autoencoder, we use the X values as input and output:
```Python 
autoencoder.fit(X_train_flat, X_train_flat,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test_flat, X_test_flat))
```
After we train this autoencoder, we can visually inspect how well it is doing. We first extract a single image from the test set. We need to add a batch dimension to this image to run it through the model, which is what we use `np.expand_dims` for. 

```Python 
original = np.expand_dims(X_test_flat[0],0)
```
Now, we run the original image through the autoencoder. The original image shows a seven, so we should hope that the output of our autoencoder shows a seven as well:
```Python 
seven = autoencoder.predict(original)
```

We now reshape both the autoencoder output as well as the original image back into 28 by 28 pixel images.
```Python 
seven = seven.reshape(1,28,28)
original = original.reshape(1,28,28)
```

We plot the original and reconstructed image next to each other. `matplotlib` does not allow the image to have a batch dimension, so we need to pass an array without it. By indexing the images with `[0,:,:]` we pass only the first item in the batch with all pixels. This first item has no batch dimension anymore.
```Python 
fig = plt.figure(figsize=(7, 10))
a=fig.add_subplot(1,2,1)
a.set_title('Original')
imgplot = plt.imshow(original[0,:,:])

b=fig.add_subplot(1,2,2)
b.set_title('Autoencoder')
imgplot = plt.imshow(seven[0,:,:])
```

![Autoencoder result](./assets/seven_autoencoded.png)

As you can see, the reconstructed seven is still a seven, so the autoencoder did manage to capture the general idea of what a seven is. It is a bit blurry around the edges, especially in the top left. It seems as the autoencoder is unsure about the length of the dashes, but it has a strongly encoded representation that there are two dashes for a seven and the general direction they follow. 

An autoencoder like this basically performs principal component analysis (PCA). It learns which components matter most for a seven to be a seven. Being able to learn this representation is useful not only for images. In credit card fraud detection for example, such principal components make for good features another classifier can work with. In the next section we will apply an autoencoder to the credit card fraud problem.

## Auto encoder for credit cards

In this section, we will once again deal with the problem of credit card fraud. This time, we will use a slightly different dataset from that in chapter 1. The new dataset contains records of actual credit card transactions with anonymized features. The dataset does not lend itself to much feature engineering. We will have to rely on end to end learning methods to build a good fraud detector. 

You can find the dataset under the following URL:
https://www.kaggle.com/mlg-ulb/creditcardfraud

And the notebook with an implementation of an autoencoder and variational autoencoder under this URL:
https://www.kaggle.com/jannesklaas/credit-vae

As usual, we first load the data. The time feature shows the absolute time of the transaction which makes it a bit hard to deal with here. So we will just drop it.
```Python 
df = pd.read_csv('../input/creditcard.csv')
df = df.drop('Time',axis=1)
``` 

We separate the X data on the tansaction from the classification of the transaction and extract the numpy array that underlies the pandas dataframe.

```Python
X = df.drop('Class',axis=1).values 
y = df['Class'].values
```

Now we need to scale the features. Feature scaling makes it easier for our model to learn a good representation of the data. This time around, we employ a slightly different method of feature scaling than before: We scale all features to be in between zero and one, as opposed to having mean zero and a standard deviation of one. This ensures that there are no very high or very low values in the dataset. But beware, that this method is susceptible to outliers influencing the result. For each column, we first subtract the minimum value, so that the new minimum value becomes zero. We then divide by the maximum value so that the new maximum value becomes one. By specifying `axis=0` we perform the scaling column wise.

```Python 
X -= X.min(axis=0)
X /= X.max(axis=0)
```

Finally, we split our data:
```Python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.1)
```

We we create the exact same autoencoder as we did before, just with different dimensions. Our input now has 29 dimensions, which we compress down to 12 dimensions before aiming to restore the original 29 dimensional output.
```Python 
from keras.models import Model
from keras.layers import Input, Dense
```

You will notice that we are using the sigmoid activation function in the end. This is only possible because we scaled the data to have values between zero and one. We are also using a tanh activation of the encoded layer. This is just a style choice that worked well in experiements and ensures that encoded values are all between minus one and one. You might use different activations functions depending on your need. If you are working with images or deeper networks, a relu activation is usually a good choice. If you are working with a more shallow network as we are doing here, a tanh activation often works well.
```Python 
data_in = Input(shape=(29,))
encoded = Dense(12,activation='tanh')(data_in)
decoded = Dense(29,activation='sigmoid')(encoded)
autoencoder = Model(data_in,decoded)
```

We use a mean squared error loss. This is a bit of an unusual choice at first, using a sigmoid activation with a mean squared error loss, yet it makes sense. Most people think that sigmoid activations have to be used with a crossentropy loss. But crossentropy loss encourages values to be either zero or one and works well for classification tasks where this is the case. But in our credit card example, most values will be around 0.5. Mean squared error is better at dealing with values where the target is not binary, but on a spectrum.
```Python
autoencoder.compile(optimizer='adam',loss='mean_squared_error')
```

After training, the autoencoder converges to a low loss. 
```Python
autoencoder.fit(X_train,
                X_train,
                epochs = 20, 
                batch_size=128, 
                validation_data=(X_test,X_test))
```

The reconstruction loss is low, but how do we know if our autoecoder is doing good? Once again, visual inspection to the rescue. Humans are very good at judging things visually, but not very good at judging abstract numbers. 

We will first make some predictions, in which we run a subset of our test set through the autoencoder. 
```Python 
pred = autoencoder.predict(X_test[0:10])
```

We can then plot indivdual samples. The code below produces an overlaid barchart comparing the original transaction data with the reconstructed transaction data.
```Python 
import matplotlib.pyplot as plt
import numpy as np

width = 0.8

prediction   = pred[9]
true_value    = X_test[9]

indices = np.arange(len(prediction))

fig = plt.figure(figsize=(10,7))

plt.bar(indices, prediction, width=width, 
        color='b', label='Predicted Value')

plt.bar([i+0.25*width for i in indices], true_value, 
        width=0.5*width, color='r', alpha=0.5, label='True Value')

plt.xticks(indices+width/2., 
           ['V{}'.format(i) for i in range(len(prediction))] )

plt.legend()

plt.show()
```

![Autoencoder results](./assets/autoencoder_results.png)

Caption: Autoncoder reconstruction vs original data.

As you can see, our model does a fine job at reconstructing the original values. The visual inspection gives more insight than the abstract number. 

```Python 
encoder = Model(data_in,encoded)
```

```Python 
enc = encoder.predict(X_test)
```

```Python 
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1,n_iter=300)
res = tsne.fit_transform(enc)
```

```Python 
fig = plt.figure(figsize=(10,7))
scatter =plt.scatter(res[:,0],res[:,1],c=y_test,cmap='coolwarm', s=0.6)
scatter.axes.get_xaxis().set_visible(False)
scatter.axes.get_yaxis().set_visible(False)
```

![Credit Auto TSNE](./assets/credit_auto_tsne.png)

# Variational Autoencoders

https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd

## MNIST Example 
```Python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
```

```Python 
batch_size = 100
original_dim = 784
latent_dim = 32
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0
``` 

```Python 
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
```


```Python 
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
```

## VAEs for end to end fraud detection
```Python 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
```

```Python 
batch_size = 100
original_dim = 29
latent_dim = 6
intermediate_dim = 16
epochs = 50
epsilon_std = 1.0
```

```Python 
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
```

```Python 
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
```

```Python 
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
```

```Python 
# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
h_decoded = decoder_h(z)

decoder_mean = Dense(original_dim)
x_decoded_mean = decoder_mean(h_decoded)
```


# Visualizing latent spaces with t-SNE
https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

# Visual question answering

# Using less data: Active Learning
General explainer
https://stackoverflow.com/questions/18944805/what-is-weakly-supervised-learning-bootstrapping

With humans
https://becominghuman.ai/accelerate-machine-learning-with-active-learning-96cea4b72fdb

Without humans 
https://shaoanlu.wordpress.com/2017/04/10/a-simple-pseudo-labeling-function-implementation-in-keras/

https://www.kaggle.com/glowingbazooka/semi-supervised-model-using-keras

https://watermark.silverchair.com/nwx106.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAZswggGXBgkqhkiG9w0BBwagggGIMIIBhAIBADCCAX0GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM39jwhp2SzNRuevyeAgEQgIIBTsa3V5rEmVzmqNsROuhfGmn5F6_JDoUINJDts1FNDjsECQRMq0iydWB72KqqA0-jezc-7880brKP_zO84TNejwhpLwSVhgcguPwxcpBWGe-Mfs-H2NtCTiLKxB_6ikxbtzQxrW1ZoOJgzS_qJrhV1lRBBLlFF7PAaACpbynl_amdusILaw8eW0HJ7KbhOQDWlCgCnJ_GnoziTl4jZff_OcCRGV60Ut5TTJSLvbWl4u2wmLAg5MdYZom8-4ilemevfV08IJaYdO-4864G8mfeXxj08SDDUay9R13wUrMP4G4UDObZlQcekVb3vtZs8Zn_TO0Io-ZBQf6LjkAHDnCb3d5ECZM8OBNh167ZcqpI81NC1trCEjUOb6kNRxBkbIHk56GZATyXYmEwbuHI_XdxY6hqvDzTnpmuBshnB_OjzBtsIohbjCKxO0jKhiex0AE

# GANs 
Keras implementations 
https://github.com/eriklindernoren/Keras-GAN

https://medium.com/jungle-book/towards-data-set-augmentation-with-gans-9dd64e9628e6

WGAN
https://www.alexirpan.com/2017/02/22/wasserstein-gan.html

