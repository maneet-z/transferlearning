#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')
get_ipython().system('unzip -uq "drive/My Drive/Cancer detection/cancer_data.zip"')


# In[ ]:


get_ipython().system('unzip -uq "train.zip" -d \'train/\'')


# In[ ]:


#!unzip -uq "train.zip" -d 'train/'


# In[ ]:


get_ipython().system('unzip -uq "test.zip" -d \'test/\'')


# In[ ]:


import cv2
from numpy.random import seed
seed(101)
import tensorflow as tf
from tensorflow import keras 
from tensorflow import set_random_seed
set_random_seed(101)
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D  
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau 
from tensorflow.keras.callbacks import ModelCheckpoint 
from sklearn.metrics import confusion_matrix
import shutil 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score  
import itertools


# In[ ]:


IMAGE_SIZE = 90  #size of the images 90x90
IMAGE_CHANNELS = 3  #colored rgb images 
SAMPLE_IMAGES = 80000  #taking 80000 images for our model

os.listdir()

print(len(os.listdir('train/')))
print(len(os.listdir('test/')))
train_labels = pd.read_csv('train_labels.csv')


# In[ ]:


train_labels.shape


# In[ ]:


#Creating a dataframe

df_labels = pd.read_csv('./train_labels.csv')
print(df_labels.shape)


# In[ ]:


# 0 With no cancer
# 1 With cancer

df_labels['label'].value_counts()


# In[ ]:


df_labels.head()


# In[ ]:


# Creating same number of samples in both the classes

# take a random sample of class 0 with size equal to num samples in class 1
df_0 = df_labels[df_labels['label'] == 0].sample(SAMPLE_IMAGES, random_state = 101)
# filter out class 1
df_1 = df_labels[df_labels['label'] == 1].sample(SAMPLE_IMAGES, random_state = 101)

# concate the dataframes
df_labels = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
# shuffle
df_labels = shuffle(df_labels)

df_labels['label'].value_counts()


# In[ ]:


# train_test_split

# stratify=y creates a balanced validation set.
y = df_labels['label']

df_train, df_val = train_test_split(df_labels, test_size=0.10, random_state=101, stratify=y)

print(df_train.shape)
print(df_val.shape)


# In[ ]:


df_train['label'].value_counts()


# In[ ]:


df_val['label'].value_counts()


# In[ ]:


# Create a new directory
base_dir = 'base_dir'
os.mkdir(base_dir)

# train_dir
train_dir = os.path.join(base_dir, 'train_dir')
os.mkdir(train_dir)

# val_dir
val_dir = os.path.join(base_dir, 'val_dir')
os.mkdir(val_dir,)

# create new folders inside train_dir
no_tumor_tissue = os.path.join(train_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(train_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)

# create new folders inside val_dir
no_tumor_tissue = os.path.join(val_dir, 'a_no_tumor_tissue')
os.mkdir(no_tumor_tissue)
has_tumor_tissue = os.path.join(val_dir, 'b_has_tumor_tissue')
os.mkdir(has_tumor_tissue)


# In[ ]:


# check that the folders have been created
os.listdir('base_dir/train_dir')


# In[ ]:


# Set the id as the index in df_data
df_labels.set_index('id', inplace=True)


# In[ ]:


# # Get a list of train and val images
train_list = list(df_train['id'])
val_list = list(df_val['id'])



# Transfer the train images

for image in train_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = df_labels.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'a_no_tumor_tissue'
    if target == 1:
        label = 'b_has_tumor_tissue'
    
    # source path to image
    src = os.path.join('train', fname)
    # destination path to image
    dst = os.path.join(train_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# Transfer the val images

for image in val_list:
    
    # the id in the csv file does not have the .tif extension therefore we add it here
    fname = image + '.tif'
    # get the label for a certain image
    target = df_labels.loc[image,'label']
    
    # these must match the folder names
    if target == 0:
        label = 'a_no_tumor_tissue'
    if target == 1:
        label = 'b_has_tumor_tissue'
    

    # source path to image
    src = os.path.join('train', fname)
    # destination path to image
    dst = os.path.join(val_dir, label, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# In[ ]:


train_path = 'base_dir/train_dir'
valid_path = 'base_dir/val_dir'
test_path = './test'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
train_batch_size = 150
val_batch_size = 150


train_steps = np.ceil(num_train_samples / train_batch_size)
val_steps = np.ceil(num_val_samples / val_batch_size)


# In[ ]:


datagen = ImageDataGenerator(rescale=1.0/255,vertical_flip = True,
                                  horizontal_flip = True,
                                  rotation_range=90,
                                  zoom_range=0.2, 
                                  width_shift_range=0.1,
                                  height_shift_range=0.1,
                                  shear_range=0.05,
                                  channel_shift_range=0.1)

train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=train_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=val_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen =ImageDataGenerator(rescale = 1./255).flow_from_directory(valid_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# In[ ]:


# calling pretrained network resnet 50
from keras.applications import ResNet50


# In[ ]:


input_tensor = Input (shape=(90, 90, 3))
dropout_fc = 0.5
conv_base = ResNet50(weights = 'imagenet', include_top = False, input_shape = (90,90,3))


# In[ ]:


model = Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))


# In[ ]:


# Resnet model summary
conv_base.summary()


# In[ ]:


# compelte model summary with dense layers attached in bottom creating bottle neck architecture.
model.summary()


# In[ ]:


# using bottom layers for training the network for better accuracy and learning complex problems
conv_base.Trainable=True

set_trainable=False
for layer in conv_base.layers:
    if layer.name == 'res5b_branch2b':
        set_trainable = True
if set_trainable:
    layer.trainable = True
else:
    layer.trainable = False


# In[ ]:


from keras import optimizers


# In[ ]:


model.compile(optimizers.Adam(0.00017), loss = "binary_crossentropy", metrics = ["acc"])


# In[ ]:


history = model.fit_generator(train_gen, steps_per_epoch=train_steps,validation_data=val_gen,validation_steps=val_steps,epochs=15,verbose = 1)


# In[ ]:


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()


# In[ ]:


# predictions of validation data
predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)


# In[ ]:


test_gen.class_indices


# In[ ]:


df_preds = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])

df_preds.head()


# In[ ]:


# Get the true labels
y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['has_tumor_tissue']


# In[ ]:


roc_auc_score(y_true, y_pred)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# In[ ]:


# Get the labels of the test images.

test_labels = test_gen.classes
test_labels.shape


# In[ ]:


# argmax returns the index of the max value in a row
cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
# Print the label associated with each class
test_gen.class_indices


# In[ ]:


# Define the labels of the class indices. These need to match the 
# order shown above.
cm_plot_labels = ['no_tumor_tissue', 'has_tumor_tissue']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# In[ ]:


from sklearn.metrics import classification_report

# Generate a classification report
# For this to work we need y_pred as binary labels not as probabilities
y_pred_binary = predictions.argmax(axis=1)
report = classification_report(y_true, y_pred_binary, target_names=cm_plot_labels)
print(report)


# In[ ]:


# create test_dir
test_dir = 'test_dir'
os.mkdir(test_dir)
    
# create test_images inside test_dir
test_images = os.path.join(test_dir, 'test_images')
os.mkdir(test_images)


# In[ ]:


# check that the directory we created exists
os.listdir('test_dir')


# In[ ]:


# Transfer the test images into image_dir

test_list = os.listdir('test')

for image in test_list:
    
    fname = image #+ '.tif'
    
    # source path to image
    src = os.path.join('test', fname)
    # destination path to image
    dst = os.path.join(test_images, fname)
    # copy the image from the source to the destination
    shutil.copyfile(src, dst)


# In[ ]:


test_path ='test_dir'


# Here we change the path to point to the test_images folder.

test_gen = datagen.flow_from_directory(test_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# In[ ]:


num_test_images = 57458

# make sure we are using the best epoch
#model.load_weights('model.h5')

predictions = model.predict_generator(test_gen, steps=num_test_images, verbose=1)


# In[ ]:


# Put the predictions into a dataframe

df_preds = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])

df_preds.head()


# In[ ]:


# This outputs the file names in the sequence in which 
# the generator processed the test images.
test_filenames = test_gen.filenames

# add the filenames to the dataframe
df_preds['file_names'] = test_filenames

df_preds.head()


# In[ ]:




