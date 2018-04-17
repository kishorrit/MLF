# Chapter 5 - Natural Language processing

It is no accident that Peter Brown, Co-CEO of Renaissance Technologies, one of the most successful quantitative hedge-funds of all time, previously worked at IBM applying machine learning to natural language problems. Information drives finance, and the most important source of information is written or spoken language. Ask any professional what they are actually spending time on and you will find that a significant part of finance is about reading. Headlines on tickers, Form-10Ks, the financial press, analyst reports, the list goes on and on. Automatically processing this information can increase speed of trades and increase the breath of information considered for trades while at the same time reducing costs.

But natural language processing (NLP) is also making inroads into finance in other areas. Insurances increasingly look to process claims automatically, retail banks try streamline their customer service and offer better products to their clients. Understanding text is increasingly becoming the go-to application of machine learning in finance.

Historically, NLP relied on hand crafted rules by linguists. Today, the linguists are getting replaced by neural networks that can learn the complex and often hard to codify rules of language. In this chapter, we will learn how to build powerful natural language models with Keras as well as how to use the SpaCy NLP library.

# A quick guide to SpaCy
SpaCy is a library for advanced natural language processing. It comes with a range of useful tools and pre-trained models that make NLP easier and more reliable. It is also pretty fast. To use SpaCy, you will need to install the library, and download its pre trained models separately:

```bash
$ pip install -U spacy

$ python -m spacy download en
```
This chapter makes use of the english language models, but more are available. Most features are available in English, German, Spanish, Portuguese, French, Italian and Dutch. Entity recognition is available for many more languages through the multi-language model.

The core of SpaCy are the `Doc` and `Vocab` classes. A `Doc` instance contains one document, including its text, tokenized version, recognized entities, etc. The `Vocab` class keeps track of all common information across documents. SpaCy is useful for its pipeline features, that contain many pieces needed for NLP. If this all seems a bit abstract right now, don't worry. This section will show you how to use SpaCy for many practical tasks. 

The data for the first section we use a collection of 143,000 articles from 15 American publications. The data is spread out over three excel files. We can load them separately, merge them into one large dataframe and then delete the individual dataframes to save memory.

```Python 
a1 = pd.read_csv('../input/articles1.csv',index_col=0)
a2 = pd.read_csv('../input/articles2.csv',index_col=0)
a3 = pd.read_csv('../input/articles3.csv',index_col=0)

df = pd.concat([a1,a2,a3])

del a1, a2, a3
```

The data looks like this:

|id|title|publication|author|date|year|month|url|content|
|--|-----|-----------|------|----|----|-----|---|-------|
|17283|House Republicans Fret...|New York Times|Carl Hulse|2016-12-31|2016.0|12.0|NaN|WASHINGTON — Congressional Republicans...|

We can plot the distribution of publishers to get an idea of what kind of news we are dealing with:
```Python 
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
df.publication.value_counts().plot(kind='bar')
```

![News Page Distribution](./assets/news_page_distribution.png)

The dataset contains no articles from classical financial news media but mostly articles from mainstream publications and politically oriented publications. 

# Named entity recognition
A common task in natural language processing is named entity recognition (NER). NER is about finding things the text explicitly refers to. Before discussing more about what is going on, lets jump right in and do some NER on the first article in our dataset.

First we need to load SpaCy as well as the model for english language processing. 
```Python 
import spacy
nlp = spacy.load('en')
```

We then select the text of the article from our data.
```Python 
text = df.loc[0,'content']
```

And now we run this piece of text through the english language model pipeline. This will create a `Doc` instance mentioned earlier. The instance holds a lot of information, including the named entities.
```Python 
doc = nlp(text)
```

SpaCy comes with a handy visualizer called `displacy` which we can use to show the named entities in text.
```Python 
from spacy import displacy
displacy.render(doc, #1
                style='ent', #2
                jupyter=True) #3
```
\#1 We pass the document.

\#2 We specify that we would like to render entities

\#3 We need to let `displacy` know that we are running this in a jupyter notebook so that rendering works correctly.

![Spacy Tags](./assets/spacy_nyt_tags.png)

And voila! As you can see, there are a few mishaps, such as blank spaces being classified as organizations or 'Obama' being classified as a place. This is because the tagging is done by a neural network and neural networks strongly dependent on the data they were trained on. You might need to fine tune the tagging model for your own purposes, and we will see in a minute how that works. You can also see that the NER offers a wide range of tags, some of which come with strange abbreviations. We will examine a full list of tags a bit later. For now, let's answer a different question: Which organizations do the news write about?

To make this exercise run faster, we will create a new pipeline in which we disable everything but NER.
```Python 
nlp = spacy.load('en',
                 disable=['parser', 
                          'tagger',
                          'textcat'])
``` 

Now we loop over the first 1000 articles from our dataset.
```Python 
from tqdm import tqdm_notebook

frames = []
for i in tqdm_notebook(range(1000)):
    doc = df.loc[i,'content'] #1
    text_id = df.loc[i,'id'] #2
    doc = nlp(doc) #3
    ents = [(e.text, e.start_char, e.end_char, e.label_) #4
            for e in doc.ents 
            if len(e.text.strip(' -—')) > 0]
    frame = pd.DataFrame(ents) #5
    frame['id'] = text_id #6
    frames.append(frame) #7
    
npf = pd.concat(frames) #8

npf.columns = ['Text','Start','Stop','Type','id'] #9   
```

\#1 We get the article content of the article at row `i`.

\#2 We get the id of the article.

\#3 We run the article through the pipeline.

\#4 For all entities found, we save the text, index of the first and last character as well as the label, but only if the tag consists of more than white spaces and dashes. This removes some of the mishaps of the classification in which empty segments or delimiters are tagged.

\#5 We create a pandas data frame out of the array of tuples created above

\#6 We add the id of the article to all records of our named entities.

\#7 We add the data frame containing all the tagged entities of one document to a list. This way we can build a collection of tagged entities over a larger number of articles.

\#7 We concatenate all data frames in the list, meaning that we create one big table with all tags.

Now we can plot the distribution of the types of entities that we found.
```Python 
npf.Type.value_counts().plot(kind='bar')
```

![Spacy Tag Distribution](./assets/spacy_tag_distribution.png)

The english language NER that comes with SpaCy is a neural network trained on the OntoNotes 5.0 corpus. It can thus recognize the following categories:

- `PERSON`: People, including fictional characters.
- `ORG`: Companies, agencies, institutions.
- `GPE`: Places including countries, cities & states.
- `DATE`: Absolute (e.g. 'January 2017') or relative dates (e.g. 'two weeks')
- `CARDINAL`: Numerals that are not covered by other types
- `NORP`: Nationalities or religious or political groups.
- `ORDINAL`: 'first', 'second', etc...
- `TIME`: Times shorter than a day (e.g. 'two hours')
- `WORK_OF_ART`: Titles of books, songs, etc.
- `LOC`: Locations that are not `GPE`s, e.g. mountain ranges or streams
- `MONEY`: Monetary values 
- `FAC`: Facilities such as airports, highways or bridges
- `PERCENT`: Percentages
- `EVENT`: Named hurricanes, battles, sporting events, etc.
- `QUANTITY`: Measurements such as weights or distance.
- `LAW`: Named documents that are laws.
- `PRODUCT`: Objects, vehicles, food, etc.
- `LANGUAGE`: Any named language. 

Next, we will have a look at the 15 most frequently named organizations:

```Python 
orgs = npf[npf.Type == 'ORG']
orgs.Text.value_counts()[:15].plot(kind='bar')
```

![Spacy Org Dist](./assets/spacy_org_distribution.png)

As you can see, political institutions such as the senate are most frequently named in our news dataset. But some companies that were in the center of media attention are found as well. Also notice how 'the White House' and 'White House' are listed as two separate organizations. Depending in your needs you might want to do some post processing such as removing 'the' from organization names. Also note how 'Trump' is shown as an organization here. If you look at the tagged text above, you will also see that 'Trump' is tagged as a NORP, speak a political organization several times. This is because the NER infers the type of tag from context. Since Trump is U.S. president, his name often gets used in the same context as (political) organizations are. 

From here, you could conduct all kinds of other investigations. The pre-trained NER gives a powerful tool that solves many common NLP tasks.

## Fine tuning the NER
https://github.com/explosion/spacy/blob/master/examples/training/train_ner.py

Many times, you will find that the pre trained NER does not do well enough on the specific kind of text that you want to work with. To solve this problem, you will have to fine tune the NER model by training it with custom data. 

Your training data should be in a form like this:
```Python 
TRAIN_DATA = [
    ('Who is Shaka Khan?', {
        'entities': [(7, 17, 'PERSON')]
    }),
    ('I like London and Berlin.', {
        'entities': [(7, 13, 'LOC'), (18, 24, 'LOC')]
    })
]
```
You provide a list of tuples of the string together with the start and end points as well as types of entities you want to tag. Data like this is usually collected through manual tagging, often on platforms like Amazon Mechanical Turk. The company behind SpaCy also made a (paid) data tagging system called prodigy which allows for efficient data collection.

Once you have collected enough data, you can either fine tune a pre trained model or initialize a completely new model. 

To load and finetune a model, use the `load()` function:
```Python 
nlp = spacy.load('en')
```

To create a new model from scratch, use the `blank` function:
```Python 
nlp = spacy.blank('en')
```
This creates an empty model ready for the english language. 

Either way, we need to get access to the NER component. If you have created a blank model, you need to create a NER pipeline component and add it to the model. If you have loaded an existing model, you can just access its existing NER.
```Python 
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
else:
    ner = nlp.get_pipe('ner')
```

Next, we need to ensure that our NER can recognize the labels we have. Imagine our data contained a new type of named entity like `ANIMAL`. With the `add_label` function we can add a label type to an NER.

```Python 
for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])
```

```Python 
import random

#1
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp._optimizer #2
    if not nlp._optimizer:
        optimizer = nlp.begin_training()
    for itn in range(5): #3
        random.shuffle(TRAIN_DATA) #4
        losses = {} #5
        for text, annotations in TRAIN_DATA: #6
            nlp.update( #7
                [text],  
                [annotations],  
                drop=0.5,  #8
                sgd=optimizer,  #9
                losses=losses) #10
        print(losses)
```
\#1 We disable all pipeline components that are not the NER by first getting a list of all components that are not the NER and then disabling them for training. 

\#2 Pre trained models come with an optimizer. If you have a blank model, you will need to create a new optimizer. Note that this also resets the model weights.

\#3 We now train for a number of epochs, in this case 5.

\#4 At the beginning of each epoch, we shuffle the training data using Pythons built in `random` module, which we imported above.

\#5 We create an empty dictionary to keep track of the losses.

\#6 Now we loop over the text and anonations in the training data.

\#7 `nlp.update` performs one forward and backward pass and updates the neural network weights. We need to supply text and annotations and the function will figure out how to train a network from it.

\#8 We can manually specify the dropout rate we want to use while training.

\#9 We pass a stochastic gradient descent optimizer that performs the model updates. Note that you can not just pass a Keras or TensorFlow optimizer here but that SpaCy has its own optimizers.

\#10 We can also pass a dictionary to write losses in which we print later to monitor progress.

Your output should look something like this: 
```
{'ner': 5.0091189558407585}
{'ner': 3.9693684224622108}
{'ner': 3.984836024903589}
{'ner': 3.457960373417813}
{'ner': 2.570318400714134}
```

# Part of speech tagging
On Tuesday, the 10th of October 2017, between 9:34 and 9:36, the Dow Jones newswire encountered a technical error that let it to post some strange headlines. One of them 'Google to buy Apple' sent apple stock up over two percent. While the algorithmic trading systems obviously failed to understand that such an acquisition would be impossible as Apple had a market capitalization of $800bn at the time and the move would likely not find regulatory approval, it is also a form of success. How did these algorithms find out who was doing what to whom? 

The answer is part of speech (POS) tagging . It allows to understand which words take over which function in a sentence and how the words relate to each other.

SpaCy comes with a handy, pre trained POS tagger:
```Python 
import spacy
from spacy import displacy
nlp = spacy.load('en')

doc = 'Google to buy Apple'
doc = nlp(doc)
displacy.render(doc,style='dep',jupyter=True, options={'distance':120})
```
Again, we load the pretrained english model and run our sentence through it. Then we use `displacy` just as we did for NER. To make the graphic fit better in a book, we set the `'distance'` option to something shorter than the default so that words get displayed closer together.

![Spacy POS](./assets/spacy_pos.png)

As you can see, the POS tagger identified 'buy' as a verb and 'Google' and 'Apple' as the the nouns in the sentence. It also identified that 'Apple' is the object the action is applied to and that 'Google' is applying the action. 

We can access this information for nouns like this:
```Python 
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)        
```

Text  |Root Text|Root dep|Root Head Text
------|--------|------|-------
Google | Google | ROOT | Google
Apple | Apple | dobj | buy

'Google' is the root of the sentence, while 'Apple' is the object of the sentence. The verb applied to 'Apple' is 'buy'. From there it is only a hard coded model of price developments under and acquisition (demand for the target stock goes up, and with it the price) and a stock lookup table to a simple event driven trading algorithm. Making these algorithm understand context and plausibility is another story however.


# Rule based matching
https://spacy.io/usage/linguistic-features#section-rule-based-matching

# Document similarity

# Topic modeling
http://nbviewer.jupyter.org/github/skipgram/modern-nlp-in-python/blob/master/executable/Modern_NLP_in_Python.ipynb 

# A text classification task
https://www.figure-eight.com/data-for-everyone/

# Preparing the data
https://blog.insightdatascience.com/how-to-solve-90-of-nlp-problems-a-step-by-step-guide-fda605278e4e 

## Tokenization

## Lemmatization

# Bag of words
https://stackoverflow.com/questions/21107505/word-count-from-a-txt-file-program

# TF-IDF
https://stevenloria.com/tf-idf/

# Word embeddings

# A quick tour of the Keras functional API
https://keras.io/getting-started/functional-api-guide/

## Debugging complex models with GraphViz
https://keras.io/visualization/

# Seq2Seq models
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

# Attention
https://github.com/philipperemy/keras-attention-mechanism/blob/master/attention_lstm.py



