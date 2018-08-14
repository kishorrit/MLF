# Loading larger image datasets into Keras
In the case of MNIST, we loaded all 60,000 images of the training set and all 10,000 images of the validation set into memory. Having all data in memory is certainly fast, but not feasible once datasets get very big. We need a way to load data as it is needed, and not all in advance. 

## A brief guide to Python Generators and yield 
The weapon of choice is a Python generator. A generator works is a function that behaves like an iterator. An iterator in Python is something that we can iterate over by calling `next`. This will give us the next item from the generator. Think of the fibonacci series for example. The fibonacci series is an infinite series in which each element is the sum of the two previous elements. So the fibonacci series goes like this: 0,1,2,3,5,8,13... Imagine we wanted to iterate over this fibonacci series and get the next element in the series whenever needed. We could implement a generator for like this:

```Python 
def fibonacci_generator():
  a = 0
  b = 1
  while True:
    yield a
    a, b = b, a + b
```
To use a generator, we first have to get the generator instance:
```Python 
fib_gen = fibonacci_generator()
```
We can then get values from the generator:
```Python 
next(fib_gen)
```
```
out: 
0
```
If we call `next` again, we get the next value of the series:
```Python 
next(fib_gen)
```
```
out: 
1
```
And the next after that:
```Python 
next(fib_gen)
```
```
out: 
1
```
Then we'd get 2, 3, 5 and so on.
Notice two things here: There is an infinite loop in the function since we use `while True`. This loop will go on forever and using an infinite loop would usually be a bad idea, however, in a generator which is supposed to generate an infinite amount of values, it is very common. Second, notice the `yield` statement. Usual Python functions use the `return` keyword. 'return' returns values and ends the function. `yield` returns values but only pauses the function. Once `next` is called again, the function keeps running until it encounters the next `yield` keyword. If a `return` keyword or the end of the function is encountered, a `StopIteration` is raised. This signals that the iterator is exhausted. Generators are often used to load images as needed. 

## The Keras image generator 
After this little theory interlude, it is time to tackle a new computer vision challenge. In the next sections we will work with the plant seedlings dataset of the university of Aarhus. See Giselsson, Thomas Mosgaard and Dyrmann, Mads and Jorgensen, Rasmus Nyholm and Jensen, Peter Kryger and Midtiby, Henrik Skov, 2017 'A Public Image Database for Benchmark of Plant Seedling Classification Algorithms' for more details on the dataset. Although plant classification is not a common problem in finance, the dataset lends itself to demonstrate many common computer vision techniques and is availeble under an open domain license. Readers who wish to test their knowledge on a more relevant dataset should take a look at the State Farm Distracted Driver dataset as well as the Planet: Understanding the Amazon from Space dataset. For more information see the exercises section of this chapter.

To follow these code samples, download the data from https://storage.googleapis.com/aibootcamp/data/plants.zip and unzip it so that it is in a folder called 'test' and 'train' in your working directory.

Keras comes with an image data generator that can load files from disk out of the box.
```Python 
from keras.preprocessing.image import ImageDataGenerator
``` 
To obtain a generator reading from files, we first have to specify the generator. Keras `ImageDataGenerator` offers a range of image augumentation tools, but we will only make use of the rescaling function. Rescaling multiplies all values in an image with a constant. Most common image formats, the color values range from 0 to 255, so we want to rescale by 1 / 255:

```Python 
imgen = ImageDataGenerator(rescale=1/255)
``` 
This however, is not yet the generator that loads the images for us. The `ImageDataGenerator` class offers a range of generators, that can be created by calling functions on it. To obtain a generator loading files we have to call `flow_from_directory`. We have to specify the directory Keras should use, the batch size we would like (32) as well as the target size the images should be resized to (150 by 150 pixels).
```Python 
generator = imgen.flow_from_directory('Segmented',
                                      batch_size=32,
                                      target_size=(150, 150))
``` 
```
out:
Found 4750 images belonging to 12 classes.
```
How did Keras find the images and how does it know which classes the images belong to? Keras generator expects the following folder structure:

- Root 
  - Class 0
    - img 
    - img 
    - ... 
  - Class 1
    - img
    - img 
    - ...
  - Class 2
    - ...
    
With Pythons `os` module we can find all the folders in `'Segmented'`

```Python 
os.listdir('Segmented')
```
```
out: 
['Small-flowered Cranesbill',
 'Sugar beet',
 'Common Chickweed',
 'Scentless Mayweed',
 'Fat Hen',
 'Common wheat',
 'Charlock',
 'Cleavers',
 'Black-grass',
 'Maize',
 'Loose Silky-bent',
 'Shepherd’s Purse']
```
And in fact, there are 12 subfolders all containing images. The Keras generator will automatically yield the targets as one-hot encoded vectors.

## Separating validation data
Before we train, we need to split our dataset into training and validation data. We will use Pythons `os` module to do the actual file moving and the `tqdm` library to keep track of the operation.

```Python 
import os
from tqdm import tqdm
```
Our data is in a folder called `'Segmented'`. We want to create a validation set, so we need to create a new folder for the validation data.

``` Python 
root_dir = 'Segmented'
target_root = 'Validation'

if not os.path.isdir(target_root):
  os.mkdir(target_root)
```

`os.path.isdir` checks if a path is an existing directory. `os.mkdir` creates a directory. Note that all paths are relative to the working directory of the notebook. Now it is time to create our validation set. To this end, we will move 12 pictures of every plant into the validation set.

```Python 
for plant in tqdm(os.listdir(root_dir)):
  plant_path = os.path.join(root_dir,plant)
  target_plant_path = os.path.join(target_root,plant)
  
  if not os.path.isdir(target_plant_path):
    os.mkdir(target_plant_path)
    
    
  files = os.listdir(plant_path)
  for i in range(12):
    source_path = os.path.join(plant_path,files[i])
    dest_path = os.path.join(target_plant_path,files[i])
    os.rename(source_path,dest_path)
``` 
`os.listdir` lists all content in a folder. So for our root folder it will list all the plant directories. By wrapping this list we are looping over in the `tqdm` function we automatically create a progress bar for our for loop. We will use `tqdm` on various occasions throughout this book and get a deeper understanding of it, but for now all you have to remember is that it is an easy way to create progress bars. Once we have a subfolder, the name of a plant, we create a source and target directory for that plant. If there is no subfolder with the plants name in our validation set we create one with `os.mkdir`. We then obtain a list of all files in the plant directory. We loop over 12 of them and move them to the validation set. Moving files in Python is done with `os.rename`. 

## Training from a Generator 
To train a model from a data generator, we need to use the `model.fit_generator` function. In this section, we will train a logistic regressor to recognize plants. Starting of with such a simple model is a good sanity check and baseline. 

We expect such a simple model to perform poorly. If it does not, we know that there is something wrong with our dataset. Sometimes, just a few pixels, or the general brightness or color of the image give away the class. A classic example of this is the story of the tanks in the forrest. The story goes that the US army wanted to train a computer vision system to detect tanks in a forest. They contracted a few researchers and supplied them with images of tanks. The researchers then went out and snapped some pictures of plain forest without tanks. Their model trained well and so they began testing their detector on real life tanks. To their surprise, it did not work. After much searching, they found out that the tank images all were taken on cloudy days while their plain forest images were taken on sunny days. Their neural network had learned to spot cloudy days, with dim and grey lightning, not tanks. Detecting dim light is much easier than detecting tanks, and to make sure that we don't have such 'giveaways' in our dataset, we should sanity check with a very simple model and hope it does poorly. This simple model will also serve as a baseline. Every more advanced model will have to show it is better than the very simple model.

To define the logistic regressor we first need to flatten the input, as the images are three dimensional blocks (height, width, 3 color channels) while logistic regression works on flat vectors.
```Python 
from keras.layers import Flatten,Dense, Activation
from keras.models import Sequential

model = Sequential()
model.add(Flatten(input_shape=(150,150,3)))
model.add(Dense(12))
model.add(Activation('softmax'))
```
Since our generator generate one hot encoded targets we can use `'categorical_crossentropy'` loss. Since this is the simple baseline we use plain stochastic gradient descent as an optimizer.
```Python 
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', 
              metrics = ['acc'])
``` 

Before we start training, we need to set up our image data generators. We can use the same basic `ImageDataGenerator` but with two different `flow_from_directory`, specifying two different paths:

```Python 
train_generator = imgen.flow_from_directory('Segmented',
                                            batch_size=32, target_size=(150,150))
                                            
```
```
out: 
Found 5515 images belonging to 12 classes.
```

```Python 
validation_generator = imgen.flow_from_directory('Validation',
                                                 batch_size=32, 
                                                 target_size=(150,150))
                                            
```
```
out: 
Found 144 images belonging to 12 classes.
```
And now to the training! To train models on generators in Keras we use `model.fit_generator`
```Python
model.fit_generator(train_generator,
                    epochs=10,
                    steps_per_epoch= 5515 // 32, 
                    validation_data=validation_generator, 
                    validation_steps= 144//32)
```

We first have to specify the generator we want to train on. Note that the generator generates data as well as targets. As usual, we also have to specify the number of epochs we want to train. What is new that we have to let keras know how many steps per epoch we want to train. Since generators never stop outputting data, how long one complete run over the dataset is. There are 5515 images in our training set and our generator yields batches of 32 samples so we need to divide 5512 by 32 to see how many times we have to call the generator to run over the data set once. Since steps can only be integer numbers, we have to use an integer division. 5515 does not cleanly divide by 32 so Keras will not do an actual complete run over the dataset, but the small inaccuracy does not matter very much. The same has to be done to specify the validation generator and the steps the validation generator has to take.

```
out:
Epoch 1/10
172/172 [==============================] - 20s 119ms/step - loss: 1.9489 - acc: 0.3598 - val_loss: 1.7644 - val_acc: 0.3984
Epoch 2/10
172/172 [==============================] - 20s 119ms/step - loss: 1.5784 - acc: 0.5182 - val_loss: 1.6448 - val_acc: 0.4688
Epoch 3/10
172/172 [==============================] - 21s 119ms/step - loss: 1.4138 - acc: 0.5751 - val_loss: 1.6180 - val_acc: 0.4531
Epoch 4/10
172/172 [==============================] - 21s 119ms/step - loss: 1.2930 - acc: 0.6238 - val_loss: 1.4567 - val_acc: 0.5156
Epoch 5/10
172/172 [==============================] - 20s 117ms/step - loss: 1.1999 - acc: 0.6509 - val_loss: 1.4917 - val_acc: 0.4844
Epoch 6/10
172/172 [==============================] - 20s 116ms/step - loss: 1.1256 - acc: 0.6824 - val_loss: 1.3618 - val_acc: 0.5391
Epoch 7/10
172/172 [==============================] - 20s 118ms/step - loss: 1.0701 - acc: 0.7043 - val_loss: 1.3135 - val_acc: 0.5781
Epoch 8/10
172/172 [==============================] - 20s 117ms/step - loss: 1.0045 - acc: 0.7278 - val_loss: 1.3783 - val_acc: 0.5625
Epoch 9/10
172/172 [==============================] - 20s 118ms/step - loss: 0.9629 - acc: 0.7388 - val_loss: 1.2864 - val_acc: 0.5938
Epoch 10/10
172/172 [==============================] - 20s 114ms/step - loss: 0.9257 - acc: 0.7542 - val_loss: 1.3282 - val_acc: 0.5469
```

Out model achieves a bit more than 54% accuracy on the validation set. Not a terribly good value, so there seems to be no giveaway features in the images. The model also overfits, since it achieves over 75% accuracy on the training set. 

# Working with pre-trained models

Training large computer vision models is hard and computationally expensive. Therefore, it is common to use models that were originally trained for another purpose and fine-tune them for a new purpose. In this section we will fine tune VGG16 originally trained on Image-Net. The Image-Net competition is an annual computer vision competition. The Image-Net dataset consists of millions of images of real world objects, from dogs to planes. Reasearchers compete to build the most accurate models. Image-Net has driven much progress in computer vision and the models built for Image-Net are a popular basis to fine tune models from. VGG16 is a model architecture developed by the visual geometry group at Oxford university. It consists of a convolutional part and a classification part. We will use the convolutional part only and add our own classification part that can classify plants. VGG16 can be downloaded via Keras:
```Python 
from keras.applications.vgg16 import VGG16
vgg_model = VGG16(include_top=False,input_shape=(150,150,3))
```
```
out: 
Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
58892288/58889256 [==============================] - 5s 0us/step
``` 

When downloading the data we want to let Keras know we do not want to include the top (the classification part), and the desired input shape. If we do not specify the input shape, the model will accept any image size but it will not be possible to add `Dense` layers on top.
```Python 
vgg_model.summary()
```
```
out: 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 150, 150, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
``` 
As you can see, the VGG model is very large with over 14 million trainable parameters. It consists of `Conv2D` and `MaxPooling2D` layers which we already learned about when working on MNIST. There are two different ways we can proceed from here: 1) Add layers and build a new model or 2) preprocess all images through the pertained model and then train a new model.

## Modifying VGG16
In this section, we will add layers on top of the VGG16 model and then train the new, big model. We do not want to retrain all those convolutional layers that have been trained already however. So we first need to 'freeze' all the layers in VGG16:
```Python 
for layer in vgg_model.layers:
  layer.trainable = False
``` 
Keras downloads VGG as a functional API model. We will learn about the functional API later, now we just want to use the Sequential API which is easier. We can convert a model to the functional API like this:
```Python 
finetune = Sequential(layers = vgg_model.layers)
``` 

This creates a new model called `finetune` which works just like a normal Sequential model now. Note that converting models to the Sequential API only works if the model can actually be expressed in the Sequential API. Some more complex models can not be converted. Adding layers to our model is now simple: 
```Python 
finetune.add(Flatten())
finetune.add(Dense(12))
finetune.add(Activation('softmax'))
```
If we look at the summary for this model we see that only our newly added layers are trainable:
```Python 
finetune.summary()
```
```
out: 

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 150, 150, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 12)                98316     
_________________________________________________________________
activation_2 (Activation)    (None, 12)                0         
=================================================================
Total params: 14,813,004
Trainable params: 98,316
Non-trainable params: 14,714,688
_________________________________________________________________
``` 
We can train this model the same way we trained the logistic regressor before.

```Python 
finetune.compile(loss='categorical_crossentropy',
                 optimizer='adam', 
                 metrics = ['acc'])
                 
finetune.fit_generator(train_generator,
                   epochs=10,
                   steps_per_epoch= 5515 // 32, 
                   validation_data=validation_generator, 
                   validation_steps= 144//32)
```
```
out: 
Epoch 1/10
172/172 [==============================] - 37s 213ms/step - loss: 0.9413 - acc: 0.6919 - val_loss: 0.6368 - val_acc: 0.8203
Epoch 2/10
172/172 [==============================] - 33s 192ms/step - loss: 0.4226 - acc: 0.8760 - val_loss: 0.6180 - val_acc: 0.7266
Epoch 3/10
172/172 [==============================] - 33s 192ms/step - loss: 0.3155 - acc: 0.9110 - val_loss: 0.4838 - val_acc: 0.8359
Epoch 4/10
172/172 [==============================] - 33s 192ms/step - loss: 0.2367 - acc: 0.9379 - val_loss: 0.4599 - val_acc: 0.8672
Epoch 5/10
172/172 [==============================] - 33s 192ms/step - loss: 0.1881 - acc: 0.9540 - val_loss: 0.4915 - val_acc: 0.8359
Epoch 6/10
172/172 [==============================] - 33s 193ms/step - loss: 0.1596 - acc: 0.9601 - val_loss: 0.4529 - val_acc: 0.8438
Epoch 7/10
172/172 [==============================] - 33s 191ms/step - loss: 0.1288 - acc: 0.9722 - val_loss: 0.4028 - val_acc: 0.8828
Epoch 8/10
172/172 [==============================] - 33s 191ms/step - loss: 0.1059 - acc: 0.9789 - val_loss: 0.4056 - val_acc: 0.8750
Epoch 9/10
172/172 [==============================] - 33s 191ms/step - loss: 0.0870 - acc: 0.9838 - val_loss: 0.3952 - val_acc: 0.8594
Epoch 10/10
172/172 [==============================] - 33s 191ms/step - loss: 0.0749 - acc: 0.9869 - val_loss: 0.3829 - val_acc: 0.8516
``` 
This model achieves 85% validation accuracy. Already a good bit better than the logistic regressor.

## Preprocessing and saving images
In the setup above, we have to run the image through the whole network every epoch, although we only want to train the top layers. This is good if we want to use image augmentation or create salience maps (both see below) but many times it is just unnecessary to run these images through the whole network time and time again. It is better to run the images through the network once and then save the output of the convolutional layer. To save the output, we will use `bcolz`, a library that can speeds up data handling through compression. To easier load and save data we have to define two functions:

```Python 
import bcolz
def save_array(fname, arr): 
  c = bcolz.carray(arr, rootdir=fname, mode='w')
  c.flush()
  
def load_array(fname): 
  return bcolz.open(fname)[:]
```  
In `save_array` we first create a new bcolz array. We specify were it should be written to. `flush` then writes the array to disk. In `load_array` we load the file and then return the elements in that file. To preprocess all files in the `'Segmented'` folder we run the following code:
```Python 
source = 'Segmented'
target = 'train_proc'

if not os.path.isdir(target):
  os.mkdir(target)

for plant in os.listdir(source):
  target_path = os.path.join(target,plant)
  source_path = os.path.join(source,plant)
  
  if not os.path.isdir(target_path):
    os.mkdir(target_path)
  
  print('Processing',plant)
  for file in tqdm(os.listdir(source_path)):
    img = cv2.imread(os.path.join(source_path,file))
    img = cv2.resize(img, (150, 150)) 
    img = np.expand_dims(img,0)
    out = vgg_model.predict(img)
    save_array(os.path.join(target_path,file), out)
```
First we specify source path and target directories. Then we loop over the folders in the source directory (the plants). If there is no folder for the plant in the target directory, we create one. For each plant, we loop over the files in the plant source path. We use OpenCV is a computer vision library which we will consider in depth in the next section. For now all you have to know is that it offers a bunch of useful tools such as reading and resizing images. Before we can feed the image into our model, we have to expand its dimensions. The model expects inputs to have the shape `(batch_size, 150, 150, 3)`. Images, when loaded from disk have dimensions `(150,150,3)`, so we need to expand the dimensionality to include a batch size dimension. After expanding dimensions, the images will have the shape `(1,150,150,3)`. Then we finally can run the image through the model by calling `predict`. We save the output using bcolz. By using `tqdm` we can keep track of how long the conversion takes.

```
out: 
Processing Common wheat
100%|██████████| 246/246 [00:04<00:00, 49.27it/s]
Processing Sugar beet
100%|██████████| 452/452 [00:09<00:00, 47.04it/s]
Processing Maize
100%|██████████| 248/248 [00:05<00:00, 46.39it/s]
Processing Common Chickweed
100%|██████████| 704/704 [00:13<00:00, 53.49it/s]
Processing Black-grass
100%|██████████| 321/321 [00:07<00:00, 41.90it/s]
Processing Loose Silky-bent
100%|██████████| 807/807 [00:16<00:00, 48.58it/s]
Processing Shepherd’s Purse
100%|██████████| 264/264 [00:05<00:00, 50.62it/s]
Processing Fat Hen
100%|██████████| 531/531 [00:10<00:00, 50.98it/s]
Processing Cleavers
100%|██████████| 336/336 [00:06<00:00, 52.02it/s]
Processing Scentless Mayweed
100%|██████████| 596/596 [00:11<00:00, 51.77it/s]
Processing Small-flowered Cranesbill
100%|██████████| 568/568 [00:11<00:00, 50.35it/s]
Processing Charlock
100%|██████████| 442/442 [00:09<00:00, 46.28it/s]
``` 
To process our validation data we only have to run the same code with `source` and `target` changed.

```Python 
source = 'Validation'
root_dir = 'validation_proc'


if not os.path.isdir(root_dir):
  os.mkdir(root_dir)

for plant in os.listdir(source):
  target_path = os.path.join(root_dir,plant)
  if not os.path.isdir(target_path):
    os.mkdir(target_path)
  source_path = os.path.join(source,plant)
  print('Processing',plant)
  for file in tqdm(os.listdir(source_path)):
    img = cv2.imread(os.path.join(source_path,file))
    img = cv2.resize(img, (150, 150)) 
    img = img / 255
    img = np.expand_dims(img,0)
    out = vgg_model.predict(img)
    save_array(os.path.join(target_path,file), out)
``` 
Note that bcolz saves arrays as directories containing multiple files. 

To train from this data now we need to write a new generator that loads the files from disk and yields the outputs of the VGG network.

```Python 
def bcz_imgen(root_dir, batch_size = 32): 
  dirs = os.listdir(root_dir)
  paths = []
  targets = []
  for dir in dirs:
    path = os.path.join(root_dir,dir)
    for file in os.listdir(path):
      paths.append(os.path.join(path,file))
      targets.append(dir)
   
  nclasses = len(np.unique(targets))
  nitems = len(targets)
  
  labelenc = LabelEncoder()
  int_targets = labelenc.fit_transform(targets)
  onehot_enc = OneHotEncoder(sparse=False)
  int_targets = int_targets.reshape(len(int_targets), 1)
  onehot_targets = onehot_enc.fit_transform(int_targets)
  
  indices = np.arange(len(paths))
  np.random.shuffle(indices)
  
  while True:
    image_stack = []
    target_stack = []
    for index in indices:
      path = paths[index]
      target = onehot_targets[index]
      img = load_array(path)

      image_stack.append(img)
      target_stack.append(target)

      if len(image_stack) == batch_size:
        images = np.concatenate(image_stack,axis=0)
        one_hot_targets = np.stack(target_stack)
        yield images, one_hot_targets
        image_stack = []
        target_stack = []
```

There is a lot going on in this code snippet, so lets break it down. The first part of the function loops over all sub directories of a a given root directory and indexes all files in the sub directories. In this case, we the root directory could be `'train_proc'` and the subdirectories would be the plant names. Since the directory a file is in indicates the class the file belongs to, the name of the directory is saved as a target. The targets are then encoded from names such as 'Maize' to integers using the sci kit learn `LabelEncoder`. Each target name is assigned an integer. This is done so because the `OneHotEncoder` accepts only integers. Using the `OneHotEncoder`, all targets get converted to one-hot. To make sure that we randomly walk over the data, we create and shuffle indices. `np.arrange` yields a sequence of integers from zero to the, but not including, length of the target. So if we have for example 5000 files, `np.arrange` yields 1,2,3,4...4998,4999. This sequence is then shuffled. After the setup is done, we create an infinite while loop. Inside this while loop we keep track of an image stack and a target stack. We loop through our random index, loading the files in the random order. Once we have a number of images and targets together, we combine them into a single numpy array and yield them. Note, that for the images we want to use `np.concatenate` to stack them along an existing axis while for the targets we want to use `np.stack` to stack them along a new axis. The images have run through the model and have a batch size dimension of 1. To create a bigger batch, we need to stack along the batch size axis. We can now define generators for training and validation: 

```Python 
train_gen = bcz_imgen('train_proc')
val_gen = bcz_imgen('validation_proc')
```
We create a new model and train it on these generators.

```Python 
model = Sequential()
model.add(Flatten(input_shape=(4,4,512)))
model.add(Dense(12))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
              
model.fit_generator(train_gen,
                    epochs=2,
                    steps_per_epoch= 5515 // 32, 
                    validation_data=val_gen, 
                    validation_steps= 144//32)
```
```
out: 
Epoch 1/2
172/172 [==============================] - 3s 15ms/step - loss: 1.6133 - acc: 0.7356 - val_loss: 3.0247 - val_acc: 0.1797
Epoch 2/2
172/172 [==============================] - 3s 17ms/step - loss: 0.3882 - acc: 0.9253 - val_loss: 3.2615 - val_acc: 0.2266
```
This model hopelessly overfits, achieving a validation accuracy of 22% at a training accuracy of 92%. However, it trained very fast due to using saved model outputs.
# Using OpenCV for image preprocessing
## Removing background
Much success in computer vision recently was about training neural nets from 'raw pixels', speak images that had not seen much image processing. However, rule based augmentation is not dead and can in fact massively boost performance of neural net based solutions if done right. In this section we will use OpenCV an numpy to remove the background of our leaves. We will do this by specifying a range of colors that plants can have, and then removing all pixels whose colors are too far away from the plant colors. To this end, we first need to define a color distance algorithm. We use a low cost approximation. 

```Python 
def colordist(img, target):
    
    img = img.astype('int')
    
    aR, aG, aB = img[:,:,0], img[:,:,1], img[:,:,2]
    bR, bG, bB = target
    
    rmean = ((aR + bR) / 2.).astype('int')
    
    r2 = np.square(aR - bR)
    g2 = np.square(aG - bG)
    b2 = np.square(aB - bB)
    
    
    result = (((512+rmean)*r2)>>8) + 4*g2 + (((767-rmean)*b2)>>8)
    
    return result
```
We use this color distance to create a mask. This mask lets us identify all pixels in an image that are not either one of the shades of leaf green. We specify the colors in brackets, (71, 86, 38) for example is the RGB code for leaf green.

```Python 
img_filter = (
  (colordist(img, (71, 86, 38)) > 1600)
  & (colordist(img, (65,  79,  19)) > 1600)
  & (colordist(img, (95,  106,  56)) > 1600)
  & (colordist(img, (56,  63,  43)) > 500)
)
```

Once we have masked an image we can remove all pixels not belonging to something leafy with 

```Python 
img[img_filter] = 0
```

The process will leave some artifacts which can be removed with OpenCV:
```Python
img = cv2.medianBlur(img, 9)
```
`medianBlur` blurs the image `img` with an aperture linear size size of 9.

This process can be rolled into a generator which we then use for training:

```Python 
def ocv_imgen(root_dir,batch_size = 32, 
              rescale = 1/255, 
              target_size = (150,150)):
              
  dirs = os.listdir(root_dir)
  paths = []
  targets = []
  for dir in dirs:
    path = os.path.join(root_dir,dir)
    for file in os.listdir(path):
      paths.append(os.path.join(path,file))
      targets.append(dir)
   
  nclasses = len(np.unique(targets))
  nitems = len(targets)
  
  labelenc = LabelEncoder()
  int_targets = labelenc.fit_transform(targets)
  onehot_enc = OneHotEncoder(sparse=False)
  int_targets = int_targets.reshape(len(int_targets), 1)
  onehot_targets = onehot_enc.fit_transform(int_targets)
  
  indices = np.arange(len(paths))
  np.random.shuffle(indices)
  while True:
    image_stack = []
    target_stack = []
    for index in indices:
      path = paths[index]
      target = onehot_targets[index]
      
      img = plt.imread(path)
      
      
      img = np.round(img * 255).astype('ubyte')[:,:,:3]
      img = cv2.resize(img, (150,150))
      img_filter = (
        (cieluv(img, (71, 86, 38)) > 1600)
        & (cieluv(img, (65,  79,  19)) > 1600)
        & (cieluv(img, (95,  106,  56)) > 1600)
        & (cieluv(img, (56,  63,  43)) > 500)
      )
      
      img[img_filter] = 0
      img = cv2.medianBlur(img, 9)
      
      image_stack.append(img)
      target_stack.append(target)
      if len(image_stack) == batch_size:
        images = np.stack(image_stack)
        images = np.divide(images,rescale)
        yield images, np.stack(target_stack)
        image_stack = []
        target_stack = []
```
This code sample works very similar to the previous generator we built ourselves. We have a setup step in which we create one-hot encoded targets and prepare our random index. However, this time we do not just load data with bcolz. For semantic reasons, we load the image with `matplotlib`. Matplotlib is a plotting library that also has some image tool. It loads image in RGB format while OpenCV loads images in BGR format. Since we want to render images in matplotlib, it is easier to also let matplotlib do the image loading. Then we apply our filters. Once we have assembled a stack we rescale it. The results looks like this: 

![BG removed](./assets/leaf_no_background.png)

You can see how everything that is not leafy green has been removed. There are a few artifacts left but the shape of the leaf is now easier to detect than it was before.

Note that rule based image augmentation can be very powerful. However, you need to make sure that the augmentation process can also be run in production. Otherwise you trained the model on something that it does not encounter in real life.

## Random image augmentation

A general problem in machine learning is that no matter how much data we have, more data would be better. More data prevents overfitting and allows our model to deal with a larger variety of inputs. It is therefore common to apply random augmentation to images, for example a rotation or a random crop. The idea is to get many different images out of one image so that the model is less likely to overfit. For many image augumentation purposes, we can just use keras `ImageDataGenerator`. More advanced augumentation can be done with OpenCV.

### Augumentation with `ImageDataGenerator`
When using an augmenting data generator we usually use it only for training. The validation generator should not use the augmentation features. The reason for this is that when we validate our model we want an estimate on how well it is doing on unseen, actual data, not augmented data. This is different from rule based augmentation where we try to create images that are easier to classify. For this reason, we need to create two `ImageDataGenerator`, one for training and one for validation. 

```Python 
train_datagen = ImageDataGenerator(
  rescale = 1/255,
  rotation_range=90,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.1,
  horizontal_flip=True,
  fill_mode='nearest')
```

This training data generator makes use of a few built in augmentation techniques. There are more available in Keras, for a full list refer to the Keras documentation. But these are commonly used:

- `rescale` scales the values in the image. We used it before and will also use it for validation. 
- `rotation_range` is a range (0 to 180 degrees) in which to randomly rotate the image. 
- `width_shift_range` and `height_shift_range` are ranges (relative to the image size, so here 20%), in which to randomly stretch images horizontally or vertically.
- `shear_range` is a range (again, relative to the image) in which to randomly apply sheer.
- `zoom_range` is the range in which to randomly zoom into a picture.
- `horizontal_flip` specifies whether to randomly flip the image. 
- `fill_mode` specifies how to fill empty spaces created by e.g. rotation. 

We can check out what the generator does by running one image through it multiple times. First, we import Keras image tools and specify an image path (this one was chosen at random).
```Python 
from keras.preprocessing import image
fname = 'train/Charlock/270209308.png'
```

We then load the image and convert it to a numpy array.
```Python 
img = image.load_img(fname, target_size=(150, 150))
img = image.img_to_array(img)
```
As before, we have to add a batch size dimension to the image:

```Python 
img = np.expand_dims(img,axis=0)
```
We then use the `ImageDataGenerator` we just created, but instead of using `flow_from_directory` we use `flow` which allows us to pass the data directly into the generator. We then pass that one image we want to use.

```Python 
gen = train_datagen.flow(img, batch_size=1)
```
In a loop, we then call `next` on our generator 4 times:
```Python 
for i in range(4):
    plt.figure(i)
    batch = next(gen)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    
plt.show()
```
![AugCharlock](./assets/aug_charlock_1.png)
![AugCharlock](./assets/aug_charlock_2.png)
![AugCharlock](./assets/aug_charlock_3.png)
![AugCharlock](./assets/aug_charlock_4.png)

### Image augmentation with OpenCV & imgaug
Need more augmentation? OpenCV offers a wide range of tools. Luckily we do not have to implement all of them ourselves. Alexander Jung, Assistant Professor of Computer Science at Aalto University has written a useful library that wraps the OpenCV tools into an easy to use augmenter. You can install it from his GitHub repository with 
```
pip install git+https://github.com/aleju/imgaug
``` 
Image augmentation is a sequential process. The filters are applied sequentially and the effect often comes from not just the use of certain filters but also the combination of filters. We can create a sequential image augmenter like this:
``` Python
from imgaug import augmenters as iaa 
seq = iaa.Sequential([
    iaa.Fliplr(0.5), 
    iaa.Crop(percent=(0, 0.1)), 
    
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    
    iaa.ContrastNormalization((0.75, 1.5)),
    
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    
], random_order=True) 

``` 
`iaa.Sequential` takes in a list of augmenters. Lets take a look at the augmenters in here:

- `Fliplr` flips images horizontally with a specified likelihood, here 50%
- `Crop` randomly crops images, here between 0% and 10%
- `Sometimes` can be wrapped around another augmenter. This augmenter is executed only with the specified probability, here `GaussianBlur` is only applied to 50% of all images.
- `GaussianBlur` adds gaussian blur with a random sigma, speak strength, between 0 and 0.5
- `ContrastNormalization` changes the contrast of the image, randomly between reducing it by a quarter and adding 50%.
- `AdditiveGaussianNoise` adds noise between 0 and 5% of the maximum pixel value (255). In 50% of all cases this is done per channel, meaning that not only the pixel brightness changes, but also the pixel color.
- `Multiply` multiplies the image values by values between 0.8, making it darker, and 1.2 making it brighter. In 20% of cases, this operation is done per channel, changing the color of the image. 

All these augmenters get passed to the sequential augmenter. Since `random_order` is set to `True`, the augmenters are not always executed in the same order, but randomly. To use this augmenter, we call `seq.augment_images(img)` where `img` is a list of images or a numpy array with a batch of images.

To try out the augmenter, we load the image as before, and then run it through the augmenter a few times.
```Python 
from keras.preprocessing import image
fname = 'train/Charlock/270209308.png'

img = image.load_img(fname, target_size=(150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)

for i in range(4):
    plt.figure(i)
    batch = seq.augment_images(img)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    
plt.show()
``` 

![OCV image aug](./assets/ocv_aug_1.png)
![OCV image aug](./assets/ocv_aug_2.png)
![OCV image aug](./assets/ocv_aug_3.png)
![OCV image aug](./assets/ocv_aug_4.png)

To use this augmenter in a generator, we can apply it in the last step of the generator we used before.

```Python 
def ocv_imgen_aug(root_dir,batch_size = 32, 
                  rescale = 1/255, 
                  target_size = (150,150)):
  
  dirs = os.listdir(root_dir)
  paths = []
  targets = []
  for dir in dirs:
    path = os.path.join(root_dir,dir)
    for file in os.listdir(path):
      paths.append(os.path.join(path,file))
      targets.append(dir)
   
  nclasses = len(np.unique(targets))
  nitems = len(targets)
  
  labelenc = LabelEncoder()
  int_targets = labelenc.fit_transform(targets)
  onehot_enc = OneHotEncoder(sparse=False)
  int_targets = int_targets.reshape(len(int_targets), 1)
  onehot_targets = onehot_enc.fit_transform(int_targets)
  
  indices = np.arange(len(paths))
  np.random.shuffle(indices)
  while True:
    image_stack = []
    target_stack = []
    for index in indices:
      path = paths[index]
      target = onehot_targets[index]
      
      img = cv2.imread(path)
      img = cv2.resize(img, target_size)
      
  
      
      image_stack.append(img)
      target_stack.append(target)
      if len(image_stack) == batch_size:
        images = np.stack(image_stack)
        
        images = seq.augment_images(images)
        
        images = np.divide(images,rescale)
        yield images, np.stack(target_stack)
        image_stack = []
        target_stack = []
```

# Understanding what ConvNets learn
Before we close this chapter, let's take a bit to think about what ConvNets learn. Neural networks in general are hard to interpret. In contrast to many classic predictive modeling techniques out of statistics or econometrics, we still lack rigorous quantitative analysis tools to figure out what exactly happens in a neural network. For computer vision applications however, there are some early promising approaches to peek inside the network. 

In this section, we will train the _input_ of a the VGG network to maximize the activations in a certain filter or network output. By finding the input that most activates a certain part of the network, we can see what the part is looking for.  

## Visualizing filters

First, we will visualize some of the convolutional filters of the network. We will use the VGG16 model again. We will have to write some advances neural network code here. We can access the backend of Keras, the tensorflow library, as a module in Keras. This makes working in tensorflow easier and we do not have to write tensorflow code directly.

```Python
import numpy as np

from keras.applications.vgg16 import VGG16
from keras import backend as K 
``` 

Next we need to load VGG16. This time, we will load the entire model.
```Python 
model = VGG16(weights='imagenet')
model.summary()
```
```
out: 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
```
Note that the layers in VGG16 have names like `block1_conv1`. We can later identify the layers we want to work with by these names. Also notice that the output of the model has 1000 dimensions. This is because the model was trained on image net, and in image net, there are 1000 classes.

From the VGG model we can create a dictionary so we can easily access the layers by name.
```Python 
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
```

Next, we have to specify our constants. We want our input image to be 224 by 224 pixels, which is the size of an input size our model expects. To begin with, we want to visualize the layer called `'block2_conv1'`. Since each convolutional layer consists of multiple convolutional filters, we also have to decide which filter we want to visualize.
```Python
img_width = 224
img_height = 224


layer_name = 'block5_conv1'
filter_index = 110
``` 

We now have to build a loss function. The loss function is the activation of the filter we want to visualize. This allows us to produce a gradient for the activation of the filter. We can then perform _gradient ascent_, which is basically the same as gradient descent, except that we want to _increase_ the loss.

```Python 
layer_output = layer_dict[layer_name].output
loss = K.mean(layer_output[:, :, :, filter_index])
```
We first get the output of the layer that we want, and then compute the mean activation of the filter we are interested in. We do this with Tensorflow so that we can get the gradients of this process in the next step. 

```Python
input_img = model.input

grads = K.gradients(loss, input_img)[0]
```
In order to compute the gradients, we first need to create a placeholder for the input. We can then get the gradients through Tensorflow. Note that we do not actually compute the gradients here yet. We merely set up the 'computational graph' through which we will run operations later. 

It has been shown that layer visualization works much better if the gradients are normalized. In order to normalize gradients, we divide them through the mean absolute value. `K.epsilon()` is a very small constant to avoid divisions by zero. Again, we are using Tensorflow to set up a graph.

```Python 
def normalize(x):
    
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())
    
grads = normalize(grads)
```

Now we come to the point were we get to use all these graphs! Since we have defined the computational graph that takes an input image and produces a loss and gradient, we can now define a function which accepts an actual immage and produces loss and gradient. The function automatically uses Tensorflow and the GPU.

```Python
iterate = K.function([input_img], [loss, grads])
```

The actual training work like this. We first initialize an image with random values:
```Python 
input_img_data = np.random.rand(1,img_height,img_width,3)
```
We set a learning rate alpha much like for gradient descent.
```Python 
alpha = 0.1
```
In a loop we then run the input image through our graph and obtain loss and gradients. We update our image along the gradients. Note that we do not update them with the negative gradients as with gradient descent in chapter one but with the positive gradients. This is called gradient ascent.
```Python 
for i in range(500):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * alpha

    print('Current loss value:', loss_value)
```

Before we can visualize the image now we first have to convert into something we can actually render, speak an RGB image. 

```Python 
def deprocess_image(x):
    
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    
    x += 0.5
    x = np.clip(x, 0, 1)

    
    x *= 255
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```
In the first step, we ensure the mean of the image data is zero and the standard deviation is 0.1. In the next step we move the mean to 0.5 and clip all values between zero and one. We then multiply all values with 255. Finally, we clip values at zero and 255 and convert them to integers. This ensures we have integer values between zero and 255 just as RGB prescribes.

Our optimized input still has a batch size dimension, we can remove this dimension by passing only the first (and only) element along this dimension in the deprocessing function.

```Python 
img = deprocess_image(input_img_data[0])
```
Finally, we can render the output image with matplotlib.

```Python 
import matplotlib.pyplot as plt

plt.style.use(['dark_background'])
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111) 

plt.imshow(img)
ax.grid(False)
```

![Conv5](./assets/conv5_vis.png)

This layer clearly responds to some pretty complex color and shape patterns. While it is hard to interpret it might be some kind of animal ear. For contrast, consider the first filter in the first layer, `'block1_conv1'`. You can render this image by running the same code only modifying two lines: 

```Python 
layer_name = 'block1_conv1'
filter_index = 0
```

![Conv1](./assets/conv1_vis.png)


This filter seems to respond too much simpler structures, diagonal lines to be precise. In general, deeper layers respond to more complex features. 
## Visualizing outputs
To visualize what triggers a certain output we can use the same method. We have to specify which output index we want to maximize. In this case we choose 184, which is the class 'Irish Terrier' in image net. 

```Python 
img_width = 224
img_height = 224

output_index = 184
```

We can use the same VGG model.
```Python 
model = VGG16(weights='imagenet')
```

Only this time our loss is not the activation of a filter but the activation of the output we want to visualize

```Python 
loss = K.mean(model.output[:, output_index])
```
Our gradient computation stays the same.

```Python
input_img = model.input
grads = K.gradients(loss, input_img)[0]
grads = normalize(grads)
iterate = K.function([input_img], [loss, grads])
```
We again start with a random image and perform gradient ascent. For output layers a smaller learning rate produces better results.

```Python
input_img_data = np.random.rand(1,img_height,img_width,3)

alpha = 0.01

for i in range(500):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * alpha
```
We can reuse our deprocessing method we used above.

```Python 
img = deprocess_image(input_img_data[0])

plt.style.use(['dark_background'])
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111) 
plt.imshow(img)
ax.grid(False)
```

![Terrier](./assets/terrier_vis.png)

This image represents the prototypical terrier in the eyes of VGG16. If you squint you can recognize some elements of a terrier. There is an eye, a nose, fluffy ears, etc. But it becomes clear that the model does not really know what a terrier _is_. It only has a statistical representation of what terriers look like. These statistical representations are why neural networks are a form of 'representation learning'. They learn the statistical representation of lines, curves, up to ears and finally the whole dog. This can also be used against them. Since this image is the ultimate representation of a terrier for the network, it could be overlaid on another image and the network would classify that image as a terrier. It has been shown that these attacks can be done with just faint overlays or a few pixels. It has also been shown that representations should be semantically interpretable for good network performance. The network should learn something that makes sense to humans.

# Exercises
1. Visit the State Farm Distracted Driver Challenge on Kaggle. Build a model that can classify distracted drivers. Can you use some rulebased preprocessing? Which augmenters aid robustness the most?
2. Use the stacked VGG model from earlier and train it on the seedlings dataset. Then visualize the outputs by backpropagating to the input. What is the ultimate representation of Maize?
3. Visit the 'Planet: Understanding the Amazon from Space' competition on Kaggle. Can you use pretrained models for statelites? Try using `model.pop()` to remove some convolutional layers from VGG16 and replace them with your own.

# Summary 
In this chapter you have learned about computer vision. From simple ConvNets on MNIST to visualizing through backprop to the input. An impressive feat! Images are still quite rarely used in finance. But an increasing number of firms incorporates image based datasources in their decision making. In the next chapter we will take a look at the most common kind of data in finance: Time series.