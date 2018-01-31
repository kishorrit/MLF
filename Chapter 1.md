# Chapter 1 - A Neural Network From Scratch

## Why Neural Networks
### Introduction
I recent years, machine learning has made great strides. Within a few years, researchers mastered tasks that where previously seen as unsolvable. From identifying objects in images, to transcribing voice, to playing complex board games, modern machine learning has matched or beaten human performance at a dazzling range of tasks. Interestingly, a single technique called 'deep learning' is behind all these advances. In fact, the bulk of of advances come from a subfield of deep learning called 'deep neural networks'. In this chapter, we will explore, how and why neural networks work.

### Some terminology
NN overview image goes here 



### Why do neural networks work?

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

Some more stuff goes here

### Gradient descent

But if a neural network can approximate an infinite number of functions, how do we find our function $f$? Researchers have come up with a method called **gradient descent**. Gradient descent tweaks the parameters


