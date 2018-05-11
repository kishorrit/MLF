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

# Visualizing latent spaces with t-SNE 
We now have a neural network that takes in a credit card transaction, and outputs a credit card transaction that looks more or less the same. But that is of course not why we built the autoecoder. The main advantage of an autoencoder is that we can now encode the transaction into a lower dimensional representation which captures the main elements of the transaction. To create the encoder model, all we have to do is to define a new Keras model, that maps from the input to the encoded state: 

```Python 
encoder = Model(data_in,encoded)
```

Note that you don't need to train this model again. The layers keep the weights from the autoencoder which we have trained before. 

To encode our data, we now use the encoder model:

```Python 
enc = encoder.predict(X_test)
```

But how would we know if these encodings contain any meaningful information about fraud? Once again, visual representation is key. While our encodings are lower dimensional than the input data, they still have twelve dimensions. It is impossible for humans to think about twelve dimensional space, so we need to draw our encodings in a lower dimensional space while still preserving the characteristics we care about. 

In our case, the characteristic we care about is _proximity_. We want points that are close to each other in the twelve dimensional space to be close to each other in the two dimensional plot. More precisely, we care about neighborhood, we want that the points that are closest to each other in the high dimensional space are also closest to each other in the low dimensional space. 

Preserving neighborhood is relevant because we want to find clusters of fraud. If we find that fraudulent transactions form a cluster in our high dimensional encodings, we can use a simple check for if a new transaction falls into the fraud cluster to flag a transaction as fraudulent. 

A popular method to project high dimensional data into low dimensional plots while preserving neighborhoods is called t-distributed stochastic neighbor embedding, or t-SNE. 

In a nutshell, t-SNE aims to faithfully represent the probability that two points are neighbors in a random sample of all points. That is, it tries to find a low dimensional representation of the data in which points in a random sample have the same probability of being closest neighbors than in the high dimensional data. 



![TSNE Info](./assets/tsne_info_one.png)
Caption: How t-SNE measures similarity

The t-SNE algorithm follows these steps:
1. Calculate the _gaussian similarity_ between all points. This is done by calculating the euclidean (spatial) distance between points and the calculate the value of a gaussian curve at that distance, see graphic. The gaussian similarity for all points $j$ from point $i$ can be calculated as:

$$p_{i|j} = \frac{exp(-||x_i-x_j||^2/2\sigma^2_i)}
{\sum_{k \neq i} exp(-||x_i-x_k||^2/2\sigma^2_i)}$$

Where $\sigma_i$ is the variance of the gaussian distribution. We will look at how to determine this variance later. Note that since the similarity between points $i$ and $j$ is scaled by the sum of distances between $i$ and all other points (expressed as $k$), the similarity between $i$ and $j$ ,$p_{i|j}$, can be different than the similarity between $j$ and $i$, $p_{j|i}$. Therefore, we average the two similarities to gain the final similarity which we work with going forward:

$$p_{ij} = \frac{p_{i|j} + p_{j,i}}{2n}$$

Where n is the number of datapoints.

2. Randomly position the data points in the lower dimensional space.

3. Calculate the _t-similarity_ between all points in the lower dimensional space. 

$$q_{ij} = \frac{(1+||y_i - y_j||^2)^{-1}}
{\sum_{k \neq l}(1+||y_k - y_l||^2)^{-1}}$$

4. Just like in training neural networks, we will optimize the positions of the data points in the lower dimensional space by following the gradient of a loss function. The loss function in this case is the Kullback–Leibler (KL) divergence between the similarities in the higher and lower dimensional space. We will give the KL divergence a closer look in the section on variational autoencoders. For now, just think of it as a way to measure the difference between two distributions. The derivative of the loss function with respect to the position $y_i$ of a datapoint $i$ in the lower dimensional space is:

$$\frac{d L}{dy_i} = 4 \sum{(p_{ij} − q_{ij})(y_i − y_j)}
(1 + ||y_i − y_j||^2)^{-1}$$


5. Adjust the data points in the lower dimensional space by using gradient descent. Moving points that were close in the high dimensional data closer together and moving points that were further away further from each other.

$$y^{(t)} = y^{(t-1)} + \frac{d L}{dy} + \alpha(t) (y^{(t-1)} - y^{(t-2)})$$

You will recognize this as a form of gradient descent with momentum, as the previous gradient is incorporated into the position update.

The t-distribution used always has one degree of freedom. The choice of one degree of freedom leads to a simpler formula as well as some nice numerical properties that lead to faster computation and more useful charts.

 The standard deviation of the gaussian distribution can be influenced by the user with a _perplexity_ hyperparameter. Perplexity can be interpreted as the number of neighbors we expect a point to have. A low perplexity value emphasizes local proximities while a large perplexity value emphasizes global perplexity values. Mathematically, perplexity can be calculated as 
 
 $$Perp(P_i) = 2^{H(P_i)}$$

Where $P_i$ is a probability distribution over the position of all data points in the dataset and $H(P_i)$ is the Shanon entropy of this distribution calculated as: 
$$H(P_i) = - \sum{p_{j|i} log_2 p_{j|i}}$$

While the details of this formula are not very relevant to using t-SNE, it is important to know that t-SNE performs a search over values of the standard deviation $\sigma$ so that it finds a global distribution $P_i$ for which the entropy over our data is our desired perplexity. In other words, you need to specify the perplexity by hand, but what that perplexity means for your dataset also depends on the dataset. 

Van Maarten and Hinton, the inventors of t-SNE, report that the algorithm is relatively robust to choices of perplexity between five and 50. The default value in most libraries is 30, which is a fine value for most datasets. If you find that your visualizations are not satisfactory, tuning the perplexity value is probably the first thing you want to do.

For all the math involved, using t-SNE is suprisingly simple. Scikit Learn has a handy t-SNE implementation which we can use just like any algorithm in scikit. We first import the `TSNE` class. Then we create a new `TSNE` instance. We define that we want to train for 5000 epochs, use the default perplexity of 30 and the default learning rate of 200. We also specify that we would like output during the training process. We then just call `fit_transform` which transforms our twelve dimensional encodings into two dimensional projections.

```Python 
from sklearn.manifold import TSNE
tsne = TSNE(verbose=1,n_iter=5000)
res = tsne.fit_transform(enc)
```

As a word of warning, t-SNE is quite slow as it needs to compute the distances between all the points. By default, sklearn uses a faster version of t-SNE called Barnes Hut approximation, which is not as precise but significantly faster already. 

There is a faster python implementation of t-SNE which can be used as a drop in replacement of sklearn's implementation. It is not as well documented however and has fewer features. You can find the faster implementation with installation instructions under the following URL:
https://github.com/DmitryUlyanov/Multicore-TSNE 

We can plot our t-SNE results as a scatter plot. For illustration, we will distinguish frauds from non frauds by color, with frauds being plotted in red and non frauds being plotted in blue. Since the actual values of t-SNE do not matter as much we will hide the axis.
```Python 
fig = plt.figure(figsize=(10,7))
scatter =plt.scatter(res[:,0],res[:,1],c=y_test, cmap='coolwarm', s=0.6)
scatter.axes.get_xaxis().set_visible(False)
scatter.axes.get_yaxis().set_visible(False)
```

![Credit Auto TSNE](./assets/credit_auto_tsne.png)

For easier spotting the cluster containing most frauds is marked with a circle. You can see that the frauds nicely separate from the rest of the transactions. Clearly, our autoencoder has found a way to distinguish frauds from genuine transaction without being given labels. This is a form of unsupervised learning. In fact, plain autoencoders perform an approximation of PCA, which is useful for unsupervised learning. In the chart you can see a few more clusters which are clearly separate from the other transactions but which are not frauds. Using autoencoders and unsupervised learning it is possible to separate and group our data in ways we did not even think about as much before. For example we might be able to cluster transactions by purchase type.

Using our autoencoder, we could now use the encoded information as features for a classifier. But even better, with only a slight modification of the autoencoder, we can generate more data that has the underlying properties of a fraud case while having different features. This is done with a variational autoencoder which we will look at in the next section.

# Variational Autoencoders

https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd

Autoencoders are basically an approximation for PCA. However, they can be extended to become generative models. Given an input, variational autoencoders (VAEs) can create encoding _distributions_. This means, that for a fraud case, the encoder would produce a distribution of possible encodings which all represent the most important characteristics of the transaction so that the decoder could turn all encodings back into the original transaction. This is useful, since it allows us to generate data about transactions. One 'problem' of fraud detection is that there are not all that many fraudulent transactions. Using a variational autoencoder, we can sample any amount of transaction encodings and train our classifier with more fraud transaction data.

How do VAEs do it? Instead of having just one compressed representation vector, a VAE has two: One for the mean encoding $\mu$ and one for the standard deviation of this encoding $\sigma$.

![VAE Scheme](./assets/vae_scheme.png)

Both mean and standard deviation are vectors just like the encoding vector we used for the vanilla autoencoder. However, to create the actual encoding we then sample by adding random noise with the standard deviation $\sigma$ to our encoding vector. 

To achieve a broad distribution of values, our network trains with a combination of two losses: The reconstruction loss you know from the vanilla autoencoder as well as a KL divergence loss between the encoding distribution and a standard gaussian distribution with a standard deviation of one.

## KL Divergence 
Kullback–Leibler divergence, or KL divergence for short, is one of the metrics machine learning inherited from information theory, just like crossentropy. It is used frequently but many struggle understanding it. 

KL divergence measures how much information is lost when a distribution $p$ is approximated with a distribution $q$.

Imagine you were working on some financial model and have collected data on returns of a security. Your financial modeling tools all assume a normal distribution of returns. The chart below shows the actual distribution of returns versus an approximation using a normal distribution. For the sake of this example, lets assume there are only discrete returns. We will cover continuous distributions later.

![Approximation vs actual](./assets/kl_divergence_dist.png)

Of course the returns in your data are not exactly normally distributed. But just how much information about returns would you loose if you did loose the approximation?

This is exactly what KL divergence is measuring. 

$$D_{KL}(p||q) = \sum_{i=1}^Np(x_i) \cdot (log\ p(x_i) - log\ q(x_i))$$

Where $p(x_i)$ and $q(x_i)$ are the probabilities that $x$, in this case the return, has some value $i$, say 5%. The formula above effectively expresses the expected difference in the logarithm of probabilities of the distribution $p$ and $q$.

$$D_{KL} = E[log\ p(x) - log\ q(x)]$$

This expected difference of log probabilities is the same as the average information lost if you approximate distribution $p$ with distribution $q$.

Since 
$$log\ a - log\ b = log\frac{a}{b}$$

KL divergence is usually written out as 

$$D_{KL}(p||q) = \sum_{i=1}^N p(x_i) \cdot log\ \frac{ p(x_i)}{q(x_i)}$$

Or in its continuous form as

$$D_{KL}(p||q) = 
\int_{-\infty}^{\infty} p(x_i) \cdot log\ \frac{ p(x_i)}{q(x_i)}$$

For variational autoencoders, we want the distribution of encodings to be a normal gaussian distribution with mean zero and a standard deviation of one.

When $p$ is substituted with the normal gaussian distribution $\mathcal{N}(0,1)$, and the approximation $q$ is a normal distribution with mean $\mu$ and standard deviation $\sigma$, $\mathcal{N}(\mu,\sigma)$, the KL divergence simplifies to

$$D_{KL} = -0.5 * (1+ log(\sigma) - \mu^2 - \sigma)$$

The partial derivatives to our mean and standard deviation vectors are therefore:

$$\frac{dD_{KL}}{d\mu} = \mu$$

and

$$\frac{dD_{KL}}{d\sigma} = -0.5 * \frac{(\sigma - 1)}{\sigma}$$


You can see that the derivative with respect to $\mu$ is zero if $\mu$ is zero and the derivative with respect to $\sigma$ is zero if $\sigma$ is one. This loss term is added to the reconstruction loss.

## MNIST Example 
Now on to our first VAE. This VAE will work with the MNIST dataset, which makes it easier to form an intuition about how VAEs work. In the next section we will build the same VAE for credit card fraud detection.

First we need to do some imports:
```Python
from keras.models import Model

from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras import metrics
```

Notice two new imports: The `Lamba` layer and the `metrics` module. The `metrics` module provides metrics, like the crossentropy loss which we will use to build our custom loss function. The `Lambda` layer allows us to use Python functions as layers, which we will use to sample from the encoding distribution. We will see just how the `Lambda` layer works in a bit, but first we need to set up the rest of the neural network.


First we define a few hyperparameters. Our data has an original dimensionality of 784, which we compress into a latent vector with 32 dimensions. Our network has an intermediate layer between the input and latent vector which has 256 dimensions. We will train for 50 epochs with a batch size of 100. 
```Python 
batch_size = 100
original_dim = 784
latent_dim = 32
intermediate_dim = 256
epochs = 50
``` 

For computational reasons, it is easier to learn the log of the standard deviation rather than the standard deviation itself. We create the first half of our network in which the input `x` maps to the intermediate layer `h`. From this layer our network splits into `z_mean` which expresses $\mu$ and `z_log_var` which expresses $log\ \sigma$.

```Python 
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
```

## Using the Lambda layer 
The `Lambda` layer wraps an arbitrary expression, speak python function, as a Keras layer. Yet there are a few requirements. For backpropagation to work, the function needs to be differentiable. After all, we want to update the network weights by the gradient of the loss. Luckily, Keras comes with a number of functions in its `backend` module which are all differentiable. Simple python math, such as `y = x + 4` is fine as well. 

Additionally, a `Lambda` function can take only one input argument. If the layer we want to create, the input is just the previous layer's output tensor. In this case, we want to create a layer with two inputs, $\mu$ and $\sigma$. So we will wrap both into a tuple which we can then take apart. Below you can see the function for sampling.

```Python 
def sampling(args):
    z_mean, z_log_var = args #1
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), 
                              mean=0.,
                              stddev=1.0) #2
    return z_mean + K.exp(z_log_var / 2) * epsilon #3
```

\#1 We take apart the input tuple and have our two input tensors.
\#2 We create a tensor containing random, normally distributed noise with a mean of zero and a standard deviation of one. The tensor has the shape as our input tensors (batch_size, latent_dim).
\#4 Finally, we multiply the random noise with our standard deviation to give it the learned standard deviation and add the learned mean. Since we are learning the log standard deviation, we have to apply the exponent function to our learned tensor. 

All these operations are differentiable since we are using Keras backend functions. Now we can turn this function into a layer and connect it to the previous two layers with one line:

```Python 
z = Lambda(sampling)([z_mean, z_log_var])
```

And voila, we got a custom layer which samples from a normal distribution described by two tensors. Keras can automatically backpropagate through this layer and train the weights of the layers before it.

Now that we have encoded our data, we need to decode it as well. We do this with two `Dense` layers.
```Python 
decoder_h = Dense(intermediate_dim, activation='relu')(z)

x_decoded = Dense(original_dim, activation='sigmoid')decoder_mean(h_decoded)
```
Our network is now complete. It encodes any MNIST image into a mean and a standard deviation tensor from which the decoding part then reconstructs the image. The only thing missing is the custom loss incentivising the network to both reconstruct images and produce a normal gaussian distribution in its encodings.

## Creating a custom loss 
The VAE loss is a combination of two losses: A reconstruction loss incentivizing the model to reconstruct its input well, and a KL divergence loss, incentivizing the model to approximate a normal gaussian distribution with its encodings. To create this combined loss, we have to calculate the two loss components separately first before combining them.

The reconstruction loss is the same loss that we applied for the vanilla autoencoder. Binary crossentropy is an appropriate loss for MNIST reconstruction. Since Keras implementation of a binary crossentropy loss already takes the mean across the batch, an operation we only want to do later, we have to scale the loss back up, so we devide it by the output dimensionality.
```Python 
reconstruction_loss = original_dim * metrics.binary_crossentropy(x, x_decoded)
```

The KL divergence loss is the simplified versions od KL divergence discussed in the section on KL divergence:
$$D_{KL} = -0.5 * (1+ log(\sigma) - \mu^2 - \sigma)$$

Expressed in Python:
```Python 
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) 
                                      - K.exp(z_log_var), axis=-1)     
```

Our final loss is then the mean of the sum of the reconstruction loss and KL divergence loss. 
```Python 
vae_loss = K.mean(reconstruction_loss + kl_loss)
```

Since we have used Keras backend for all calculations, the resulting loss is a tensor which can be automatically differentiated. 

Now we create our model like usual:

```Python 
vae = Model(x, x_decoded)
```

Since we use a custom loss, we have the loss separately, and can't just add it in the compile statement:
```Python 
vae.add_loss(vae_loss)
```
Now we compile the model. Since our model already has a loss, we only have to specify the optimizer. 
```Python 
vae.compile(optimizer='rmsprop')
```

Another side effect of the custom loss is that it compares the output of the VAE with the _input_ of the VAE, which makes sense as we want to reconstruct the input. Therefore we do not have to specify y values, only specifying an input is enough.
```Python 
vae.fit(X_train_flat,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_flat, None))
```

## Using a VAE to generate data 
So we got our autoencoder, how do we generate more data? We take an input, say a picture of a seven, and run it through the autoencoder multiple times. Since the autoencoder is randomly sampling from a distribution, the output will be slightly different at each run.

From our test data, we take a seven.
```Python 
one_seven = X_test_flat[0]
```

We add a batch dimension and repeat the seven across the batch four times. Now we have a batch of four, identical sevens.
```Python 
one_seven = np.expand_dims(one_seven,0)
one_seven = one_seven.repeat(4,axis=0)
```

We make a prediction on that batch. We get back the reconstructed sevens.
```Python 
s = vae.predict(one_seven)
```

We now reshape all the sevens back into image form.
```Python 
s= s.reshape(4,28,28)
```

And now we plot them:
```Python 
fig=plt.figure(figsize=(8, 8))
columns = 2
rows = 2
for i in range(1, columns*rows +1):
    img = s[i-1]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()
```

![Many sevens](./assets/vae_mult_sevens.png)

As you can see, all images show a seven. They look quite similar, but if you look closely you see there are distinct differences. The seven on the top left has a less pronounced stroke than the seven on the bottom left. The seven on the bottom right has a sight bow at the end. 

The VAE has created new data. Using this data for more training is not as good as using completely new real world data, but it is still very useful. While generative models like this one are nice for eye candy, we will now discuss how this technique can be used for credit card fraud detection.

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

