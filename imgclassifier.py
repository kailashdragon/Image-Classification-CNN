from cv2 import cv2
import os
import numpy as np
from random import shuffle
from tqdm import tqdm


train_dir = 'C:/Users/Lenovo/Downloads/train'
test_dir = 'C:/Users/Lenovo/Downloads/test1'

img_size = 50
lr = 1e-3

model_name = 'dogsvscats-{}-{}.model'.format(lr, '6conv-basic')

def label_image(img):
    word_label = img.split('.')[-3]
    if word_label=='cat':return [1, 0]
    elif word_label=='dog':return [0, 1]

def create_training_data():
    training_data = []
    
    for img in tqdm(os.listdir(train_dir)):
        label = label_image(img)
        path = os.path.join(train_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        training_data.append([np.array(img), np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    
    for img in tqdm(os.listdir(test_dir)):
        path = os.path.join(test_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_training_data()
test_data = process_test_data()

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

tf.reset_default_graph()
convnet = input_data(shape =[None, img_size, img_size, 1], name ='input')

convnet = conv_2d(convnet, 32, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = conv_2d(convnet, 64, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = conv_2d(convnet, 128, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = conv_2d(convnet, 64, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = conv_2d(convnet, 32, 5, activation ='relu')
convnet = max_pool_2d(convnet, 5)
 
convnet = fully_connected(convnet, 1024, activation ='relu')
convnet = dropout(convnet, 0.8)
 
convnet = fully_connected(convnet, 2, activation ='softmax')
convnet = regression(convnet, optimizer ='adam', learning_rate = lr,
      loss ='categorical_crossentropy', name ='targets')

model = tflearn.DNN(convnet, tensorboard_dir ='log')

train = train_data[:-10]
test = train_data[-10:]
 
X = np.array([i[0] for i in train]).reshape(-1, img_size, img_size, 1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1, img_size, img_size, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch = 5, 
    validation_set =({'input': test_x}, {'targets': test_y}), 
    snapshot_step = 500, show_metric = True, run_id = model_name)
model.save(model_name)

import matplotlib.pyplot as plt

test_data = np.load('test_data.npy')
 
fig = plt.figure()
 
for num, data in enumerate(test_data[:20]):
    # cat: [1, 0]
    # dog: [0, 1]
     
    img_num = data[1]
    img_data = data[0]
     
    y = fig.add_subplot(4, 5, num + 1)
    orig = img_data
    data = img_data.reshape(img_size, img_size, 1)
 
    # model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
     
    if np.argmax(model_out) == 1: str_label ='Dog'
    else: str_label ='Cat'
         
    y.imshow(orig, cmap ='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


        
        
