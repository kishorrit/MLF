# Chapter 8 Debugging & Deployment
After the last seven chapters, you now have a large toolbox of machine learning algorithms you could use for your problem. But what if it does not work? Machine learning models fail in the worst way: They fail silently. In traditional software, a mistake usually leads to a crash in the program. While crashes are annoying for the user, they are helpful for the programmer. At least it is clear that the code failed and when it failed. Often, there even is a crash report that describes what went wrong. Sometimes, machine learning code crashes too, for example if the data we feed in has the wrong format or shape. These issues can usually be debugged by carefully tracking which shape the data had at what point. More often however, models that fail just output poor predictions. They give no signal that they have failed and you might not be aware that they failed at all. At other times, they might not train well, won't converge or won't achieve a low loss. This chapter is all about how you debug these silent fails.

The first step is to acknowledge that even good machine learning engineers fail frequently. There are many reasons why ML projects fail, and most have nothing to do with the skills of the engineers. But engineers can be on the watch for factors that often cause project failure. If spotted early, time and money can be saved. Even more, in high stakes environments, such as trading, aware engineers can pull the plug when they notice their model is failing. This should not be seen as a failure, but as a success to avoid problems.

# Debugging data
The first chapter of this book describes that models are a function of their training data. Bad data leads to bad models. Garbage in, garbage out. If your project is failing, your data is the most likely culprit. But even if you have a working model, the real world data coming in might not be up for the task. In this section we will learn how to find out if you have good data, what to do if you have been given not enough data, and how to test your data.

## How to find out if your data is up to the task

There is two aspects to knowing if your data is up to the task of training a good model: Does the data predict what you want to predict and do you have enough data. 

To find out if your model does contain predicting information, also called a signal, you can ask if a human could make a prediction given this data. This works well for data for which you have humans making predictions already. After all, the only reason we know intelligence is possible is because we observe it in humans. Humans are good at understanding written text, but if a human does not understand a text, chances are that your model won't make much sense of it either. A common pitfall to this test is that humans have context your model does not have. A human trader does not only consume financial data but might also have experienced the product of a company or seen the CEO on TV. This context flows into the traders decision, but is often forgotten when a model is built. Humans are also good at focusing on important data. A human trader will not consume all financial data there is, because most of it is irrelevant. Adding more inputs to your model won't make it better. It often makes it worse as the model overfits and gets distracted by the noise. On the other hand, humans are irrational, follow peer pressure and have a hard time making decisions in abstract and unfamiliar environments. Humans would struggle to find an optimal traffic light policy for instance, since the data that traffic lights operate on is not intuitive to us.

This brings us to the second sanity check: A human might not be able to make predictions, but there might be a causal (economic) rationale. There is a causal link between a company's profits and its share price, the traffic on a road and traffic jams, customer complaints and leaving customers and so on. And while humans might not have an intuitive gasp on these links, we can discover them by reasoning. There are some tasks, for which a causal link is required. For a long time, many quantitative trading firms insisted on their data having a causal link to the predicted outcomes of models for instance. Nowadays, the industry seems to have moved a bit away from that as it gets more confident in testing its algorithms. 

If humans can not make a prediction and there is no causal rationale for why your data is predictive, you might want to reconsider if your project is feasible. 

Once you have determined that your data contains enough signal, you need to ask yourself if you have enough data to train a model to extract the signal. There is no clear answer to how much is enough. Roughly, the amount needed depends on the complexity of the model you hope to create. There are a couple rules of thumb to follow however:

- For classification, you should have around 30 independent samples per class.
- You should have 10 times as many samples as there are features, especially for structured data problems.
- Your dataset should get bigger as the number of parameters in your model gets bigger.

Keep in mind these rules are only rules of thumb and might be very different for your specific application. If you can make use of transfer learning, you can drastically reduce the number of samples you need. This is why most computer vision applications use transfer learning. 

If you have any reasonable amount of data, say a few hundred samples, you can start building your model. Perhaps start with a simple model which you can deploy while you collect more data.

## What to do if you don't have enough data
Sometimes, you find yourself in a situation where you simply do not have enough data. Sometimes, this happens after you already begun your project. For example, the legal team might have changed its mind and decided that you can not use the data even though they green lit it earlier. In this case, you have multiple options:

Most of the time, you can **augment your data**. You have seen some data augmentation in chapter 3. Of course, you can augment all kinds of data. For example, you could slightly change some database entries. 

Taking augmentation a step further, you might be able to **generate your data**, for example in simulation. This is effectively how most reinforcement learning research gathers data. But it also works in other cases. The data we used for fraud detection in chapter two was obtained from simulation. Simulation requires you to be able to write down the rules of your environment in a program. Powerful learning algorithms tend to figure out these, often over simplistic, rules, so they might not generalize to the real world as well. Yet, simulated data can be a powerful addition to real data.

Often, you can **find external data**. Just because you have not tracked a certain datapoint, it does not mean that nobody else has. There is an astonishing amount of data available on the internet. Even if the data was not originally collected for your purpose, you can often retool data by either relabeling it or by using it for **transfer learning**. You might be able to train a model on a large dataset for a different task and then use that model as a basis for your task. Equally, you can find a model someone else has trained for a different task, and repurpose it.

Finally, you might be able to create a **simple model**, that does not capture the relationship in the data completely but is enough to ship a product. Random forests and other tree based methods often require much less data than neural networks. 

Remember, that for data, quality trumps quantity in the majority of cases. Getting a small, high quality dataset in and training a weak model is often your best shot to find problems with data early. You can always scale up data collection later. A mistake many practitioners make is that they spend huge amounts of time and money on getting a big dataset, only to find that they have the wrong kind of data.

## Unit testing data
If you build a model, you make assumptions about your data. For example, you assume that the data you feed into your time series model is actually a time series with dates that follow each other in order. You need to test your data to make sure this assumption is true. Especially live data that you receive once your model is already in production. Bad data might lead to poor model performance, which can be dangerous especially in a high stakes environment.

Additionally, you need to test if your data is clean from things like personal information. If you buy data from a vendor and the vendor forgot to delete social security numbers from the dataset, you might still be on the hook for using peoples data without consent.

Since monitoring data quality is important when trading based on many data sources, Two Sigma, a hedge-fund, has created and open sourced a library for data monitoring. It is called marbles, see https://github.com/twosigma/marbles and builds on Pythons `unittest` library. You can install it with 

```
pip install marbles
```

You can not run unit tests on Kaggle notebooks, so you need to install marbles and all dependencies like `pandas` or `numpy` on your local machine to try this example. You can find the example code as `7_marbles_test.py` in the GitHub repository of this book.

```Python 
import marbles.core #1
from marbles.mixins import mixins #2

import pandas as pd #3
import numpy as np
from datetime import datetime, timedelta

class AgeTestCase(marbles.core.TestCase,mixins.DateTimeMixins): #4
    def setUp(self): #5
        self.df = pd.DataFrame({'parent':[datetime(1959,1,1)],
                                'child':[datetime(2001,1,1)]},
                                index=[0]) #6
        
    def tearDown(self): #7
        self.df = None
        
    def test_parent_older_child(self): #8
        self.assertDateTimesBefore(sequence=self.df.parent,
                                  target=self.df.child,
                                  note='Parents have to be born after \
                                  their children') #9
        
    def test_old_age(self): #10
        max_td = timedelta(365*100) #11
        today = datetime.today()
        
        self.assertDateTimesAfter(sequence=self.df.child, #12
                                  target=today - max_td)
        
if __name__ == '__main__':       
    marbles.core.main() #13
```

```
FF
======================================================================
FAIL: test_old_age (7_marbles_test.AgeTestCase)
----------------------------------------------------------------------
marbles.core.marbles.ContextualAssertionError: 0   1800-01-01
Name: child, dtype: datetime64[ns] is not strictly greater than 1918-06-24 22:13:09.499508

Source (/Users/jannes/Desktop/mlfin_code/7_marbles_test.py):
     25 
 >   26 self.assertDateTimesAfter(sequence=self.df.child,target=today-max_td)
     27 
Locals:
	today=2018-05-30 22:13:09.499508
	max_td=36500 days, 0:00:00


======================================================================
FAIL: test_parent_older_child (7_marbles_test.AgeTestCase)
----------------------------------------------------------------------
marbles.core.marbles.ContextualAssertionError: 0   1959-07-12
Name: parent, dtype: datetime64[ns] is not strictly less than 0   1800-01-01
Name: child, dtype: datetime64[ns]

Source (/Users/jannes/Desktop/mlfin_code/7_marbles_test.py):
     17 def test_parent_older_child(self):
 >   18     self.assertDateTimesBefore(sequence=self.df.parent,
     19                                 target=self.df.child,
     20                                 note='Parents have to be born after their children')
     21 
Locals:

Note:
	Parents have to be born after their children


----------------------------------------------------------------------
Ran 2 tests in 0.008s

FAILED (failures=2)

```

## Preparing data for training

# Your model is not right 

## Learning rate scheduling
Cosine annealing
With restarts

## Snapshot ensembles
https://github.com/titu1994/Snapshot-Ensembles

# You are solving the wrong problem 


# Debugging Overview
https://medium.com/machine-learning-world/how-to-debug-neural-networks-manual-dc2a200f10f2

https://engineering.semantics3.com/debugging-neural-networks-a-checklist-ca52e11151ec

https://www.quora.com/How-do-I-debug-an-artificial-neural-network-algorithm

https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607

https://medium.com/@skyetetra/so-your-data-science-project-isnt-working-7bf57e3f12f1

# Interpretability
https://github.com/marcotcr/lime


# Deployment 

# Testing Data 
https://marbles.readthedocs.io/en/stable/


