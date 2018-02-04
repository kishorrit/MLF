# Chapter 1 - A Neural Network From Scratch

## Why Neural Networks
### Introduction
I recent years, machine learning has made great strides. Within a few years, researchers mastered tasks that where previously seen as unsolvable. From identifying objects in images, to transcribing voice, to playing complex board games, modern machine learning has matched or beaten human performance at a dazzling range of tasks. Interestingly, a single technique called 'deep learning' is behind all these advances. In fact, the bulk of of advances come from a subfield of deep learning called 'deep neural networks'. In this chapter, we will explore, how and why neural networks work.

### Approximating functions

There are many views on how to best think about neural networks (NNs), but perhaps the most useful is to see them as **function approximators**. Functions in math relate some input $x$ to some output $y$. We can write it as:

$$y = f(x)$$

A simple function could be:

$$f(x) = 4 * x$$

In this case, we can give the function an input $x$ and it would quadruple it:

$$y = f(2) = 8$$

You might have seen functions like this in school. But functions can do more. They can map any element from a set (speak: the collection of values the function accepts) to another element of a set. These sets can be something other than simple numbers. A function could for example also map an image to an identification of what is in the image: 

$$imageContent = f(image)$$

This function would map an image of a cat to the label 'cat'.

Note that for a computer, images are matrices full of numbers and that any description of an image content would also be stored as a matrix of numbers. 

A neural network, if it is big enough, can approximate any function. This means, that a neural network, if big enough, could also approximate our function $f$ for mapping images to their content. The condition that the neural network has to be 'big enough' explains why 'deep' (speak: big) neural networks have taken off. 

The fact that 'big enough' neural networks can approximate any function means that they are useful for a large number of tasks.

### A forward pass

For the rest of the chapter we will be working with a simple problem: Given an input vector $X$ we want to output the first element of the vector. We already know we need data to train a neural network so this will be our dataset for the exercise:

|$X_1$|$X_2$|$X_3$|$y$|
|-|-|-|---|
|0|1|0|0|
|1|0|0|1|
|1|1|1|1|
|0|1|1|0|

In this dataset, each row contains an input vector $X$ and an output $y$.

The data follows the formula: 
$$ y = X_1$$
So the function we want to approximate is:
$$ f(X) = X_1$$

In this case, writing down the function is relatively straightforward but keep in mind that in most cases it is not possible to write down the function, as functions expressed by deep neural nets can get very complex.

For this simple function, a shallow neural network with only one layer is enough. Such shallow networks are also called logistic regressors.
#### A logistic regressor 
The graphic below shows a logistic regressor. $X$ is our input vector, here shown as it's three components, $X_1$, $X_2$, $X_3$. $W$ is a vector of three weights. You can imagine it as the thickness of each of the three lines. $W$ determines how much each of the values of $X$ goes into the the next layer. $b$ is the bias. It can move the output of the layer up or down.

![Log Regressor](./assets/logistic_regression.png)

To compute the output of the regressor, we first do a **linear step**. We compute the dot product of the input $X$ and the weights $W$. This is the same as multiplying each value of $X$ with it's weight and then taking the sum. To this number, we add the bias $b$. Afterwards, we do a **non linear step**. In the non linear step, we run the linear intermediate product $z$ through an **activation function**, in this case, the sigmoid function. The sigmoid function squishes input values to outputs between zero and one.

SIGMOID GRAPHIC GOES HERE 

#### Python version of our logistic regressor

If all the math above was a bit too theoretical for you, rejoice! We will now implement the same in Python.

We will use a library called numpy which enables easy and fast matrix operations in Python. To ensure we get the same result in all of our experiments, we have to set a random seed.

```python
import numpy as np
np.random.seed(1)
```

Since our dataset is quite small, we define it manually as numpy matrices.

```Python
X = np.array([[0,1,0],
              [1,0,0],
              [1,1,1],
              [0,1,1]])

y = np.array([[0,1,1,0]])
```

We can define the sigmoid activation function as a Python function.

```python
def sigmoid(x):
    return 1/(1+np.exp(-x))    
```

So far, so good. Now we need to initialize $W$. In this case, we actually know already which values $W$ should have. But we can not know for other problems where we do not know the function yet. So we have to assign weights randomly. The weights are usually assigned randomly with a mean of zero. The bias is usually set to zero by default
```python 
W = 2*np.random.random((3,1)) - 1
b = 0
```

Now that all variables are set, we can do the linear step:

```python
z = X.dot(W) + b
```
And the non linear step.
```python 
A = sigmoid(z)
```

If we print out $A$ now, we get the following output:
```Python
print(A)
```
```
out:
[[ 0.60841366]
 [ 0.45860596]
 [ 0.3262757 ]
 [ 0.36375058]]
```
This looks nothing like our desired output $y$ at all! Clearly, our regressor is representing _some_ function, but it is quite far away from the function we want. To better approximate our desired function, we have to tweak the weights $W$ and the bias $b$ to get better results.

### Gradient descent

We already saw that we need to tweak the weights and biases, collectively called parameters, of our model to arrive at a closer approximation of our desired function. In other words, we need to look through the space of possible functions that can be represented by our model to find a function $\hat f$ that matches our desired function $f$ as close as possible.


