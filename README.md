# ML Fin Book Repo

Chapters will be uploaded as Markdown Files (can be converted to word with [Pandoc](https://pandoc.org/)) or Jupyter Notebooks for chapters that contain lot's of code (can also be converted).

# Outline
## Audience
The target reader of this book is a professional working in the financial industry (a domain expert, see Mission). This includes everything from consumer facing banks, to hedge-funds to supporting services such as auditors and accountants. The book assumes basic knowledge of linear algebra and calculus, as well as some basic knowledge of Python. As a rule of thumb, readers who have completed the first two free Python courses by DataCamp should be fine. The book also assumes basic knowledge of micro economics and a rough overview of the financial industry.

The target reader has noticed that the industry is being transformed by machine learning. The reader started learning a bit of Python earlier and has maybe even dabbeled in some other machine learning courses but is unsure of how ML could have an impact in the industry and which skills could be especially valuable. The reader buys this book to get some working knowledge of the most important technologies and find out where to get deeper into.

## Mission
To better understand the mission and advantage of this book it helps to understand how ML goes from research to value at scale. In terms of business strategy, ML is a horizontal. It is part of multiple value chains across industries. Research so far has focused on strengthening this horizontal and make ML useful for many applications. However, to extract maximum value, ML must be deeply integrated into the different value chains. This will require adopted methods and approaches for each value chain. The task to adopt general ML techniques falls to ML experts and domain experts alike. Domain experts will have to use their knowledge of their domain (in this case the different parts of the financial industry) and knowledge of ML to create custom ML systems. 

This book teaches deep learning techniques that are useful in the financial industry in a way that is understandable for domain experts. It uses more of the terminology and tools known to industry professionals and makes use of concepts that are known to financial professionals.

The book takes readers on the journey from a high level understanding of what deep neural networks are, why they work and their limitations to how to practically implement state of the art techniques.

## Objectives and achievements
After reading this book readers will understand:
- Neural networks as function approximators
- The gradient descent optimization algorithm
- The value of predictive models to business
- Working with structured data
- Working with image Data
- Working with time series 
- Natural language processing 
- GANs and other generative models 
- Reinforcement learning 
- Practicalities of model deployment
- Dealing with ethical and legal concerns around deep learning 

## General structure 
### 1 - Theoretical considerations 
1. - A Neural Net from Scratch

### 2 - Practical implementations for various tasks
2. - Structured Data: From Telemarketing to Fraud Prevention
3. - Deep Learning for Visual Data: From Customer Cards to Searching Mines from Space
4. - From Engines to Stocks: Time series data
5. - Software Robots Reading Reports for you: Natural Language Processing
6. - Generative models for Financial Information Extraction
7. - Reinforcement learning for the Markets

### 3 - Deployment considerations 
8. - Debugging Neural Nets
9. - Combating Bias in Financial Models

## Detailed outline 
### A Neural Net from Scratch
**Description:** This chapter introduces neural networks as function approximators and gradient descent as a way to find an optimal function. It discusses the flaws of gradient descent and why it is useful in many applications. It concludes with an implementation of a neural network and gradient descent in Python and Excel. 

**Level:** This is a slightly more advanced and complex chapter. 

**Topics Covered:**
- Neural nets as function approximators
- Gradient descent optimization
- Feed forward
- Back propagation

**Skills learned:** Understanding what neural networks are, how and why they work on the inside.

### Structured Data: From Telemarketing to Fraud Prevention
**Description:** This chapter introduces the Keras sequential model, and how to use neural networks with structured data. 

**Level:** This is an easy chapter

**Topics Covered:**
- The Keras sequential model 
- Dense layers
- Activation layers 
- Data preparation for neural networks
- ``model.fit()``
- Some different optimizers
- Regularization
- Entity embeddings

**Skills learned:** Build and train simple models with Keras.

### Deep Learning for Visual Data: From Customer Cards to Searching Mines from Space
**Description:** This chapter introduces convolutional neural networks and how to work with them in Keras.

**Level:** This is an average chapter

**Topics Covered:**
- What are convnets and why should we use them
- ``Conv2D`` 
- ``MaxPool2D``
- Data generators
- Working with pre-trained models
- Batchnorm
- Object detection & some other advanced architectures

**Skills learned:** Build and train SOTA image classifiers.

### From Engines to Stocks: Time series data
**Description:** This chapter introduces time series models.

**Level:** This is a slightly advanced chapter

**Topics Covered:**
- RNNs
- LSTMS
- Recurrent dropout
- ``Conv1D``
- ``MaxPool1D``
- Sequence classification
- Seq2Seq

**Skills learned:** Build and train models to forecast and classify from time series.

### Software Robots Reading Reports for you: Natural Language Processing
**Description:** This chapter introduces natural language processing.

**Level:** This is an average chapter

**Topics Covered:**
- Word / Character / N-gram embeddings
- Pre-trained word embeddings: Word2Vec, GloVe
- Preparing text data 
- Text classification & Sentiment analysis
- Seq2Seq translation
- Seq2Seq summarization
- Question answering (Note: Only if I find an easy demo implementation somewhere that does not blow the difficulty level)

**Skills learned:** Build a classification & Seq2Seq NLP systems systems

### Generative models for Financial Information Extraction
**Description:** This chapter shows how to use generative models to extract useful information.

**Level:** This is an advanced chapter

**Topics Covered:**
- Autoencoders for unsupervised learning 
- Pix2Pix
- Pix2Code 
- GANs for semi supervised learning

**Skills learned:** Build an auto encoder! Translate satellite images to street maps. Generate Website code from screenshots. Create artificial training data. Hussah generative models are fun.

### Reinforcement learning for the Markets
**Description:** This chapter introduces reinforcement learning and shows some potential paths to 'AI'

**Level:** This is an advanced chapter

**Topics Covered:**
- Everything is a markov model 
- Q-Learning 
- Policy gradients
- Neuro-evolution
- On the importance of good simulators

**Skills learned:** Build systems that: Play pong. Play atari games. Walk shake and wiggle. Invent trading strategies (and why that might not work)

### Debugging Neural Nets
**Description:** This chapter shows some practicalities of deployment and debugging

**Level:** This is an easy chapter

**Topics Covered:**
- Fighting over / underfit
- Vanishing gradients
- Exploding gradients 
- Tensorboard 
- Tips on training _big_ models.
- Serving models 
- Monitoring model performance

**Skills learned:** Reduce over/underfit, monitor training with Tensorboard, create custom Keras callbacks, serve Keras with Flask, monitor models and early warning systems.

### Combating Bias in Financial Models
**Description:** This chapter discusses ideas about making ML models fair and accountable

**Level:** This is an easy chapter

**Topics Covered:**
- Why models amplify biases in data 
- Why 'de biasing' data is harder than you think
- Some examples of biased models
- FAT ML research 
- Research on black box reasoning
- Why secret models are dangerous but common + alternative strategic approaches.
- Legal situation of black box decision making (EU / US)

**Skills learned:** Thinking about discrimination issues before building models

# Author BIO
Jannes is the lead developer of the Bletchley Bootcamp (see ai-bootcamp.org), a course teaching deep learning to business and economics majors.

## Schedule
1
A Neural Net from Scratch
On or before
23/02/2018


2
Structured Data: From Telemarketing to Fraud Prevention
On or before
07/03/2018


3
Deep Learning for Visual Data: From Customer Cards to Searching Mines from Space
On or before
26/03/2018


4
From Engines to Stocks: Time series data
On or before
07/04/2018


5
Software Robots Reading Reports for you: Natural Language Processing
On or before
19/04/2018


6
Generative models for Financial Information Extraction
On or before
31/04/2018


7
Reinforcement learning for the Markets
On or before
12/05/2018


8
Debugging Neural Nets
On or before
24/05/2018


9
Combating Bias in Financial Models
On or before
05/06/2018

## Have a suggestion?
File an issue

## Want to add something?
[Fork, edit, send a pull request](https://guides.github.com/introduction/flow/). We don't even need separate branches.
