import sys
sys.path.append('../utils')
sys.path.append('../configs')

from sklearn.model_selection import train_test_split
# ../utils
from reader import read_pkl
from writer import write_npy, create_muldir
from dataset_op import label_count

# ../configs
from path import CIFAR100PATH, CIFARPROCESSED
from info import CIFARNCLASS

import numpy as np
import pickle


train = read_pkl(CIFAR100PATH+'train', encoding='bytes')
test = read_pkl(CIFAR100PATH+'test', encoding='bytes')

train_image = train[b'data'].astype(np.float32)/255
test_image = test[b'data'].astype(np.float32)/255

pixel_mean = np.mean(train_image, axis=0) # use global pixel mean only for train data

train_image -= pixel_mean
test_image -= pixel_mean

train_image = np.transpose(np.reshape(train_image, [-1,3,32,32]), [0,2,3,1])
train_label = np.array(train[b'fine_labels'])

test_image = np.transpose(np.reshape(test_image, [-1,3,32,32]), [0,2,3,1])
test_label = np.array(test[b'fine_labels'])

train_image, val_image, train_label, val_label = train_test_split(train_image, train_label, test_size=0.1, random_state=40, stratify=train_label) # Train Val split

print("Total of classes : %d"%CIFARNCLASS)
print("Train info")
print("Image : {}({}), Label : {}".format(train_image.shape, train_image.dtype, train_label.shape))
print("Val info")
print("Image : {}({}), Label : {}".format(val_image.shape, val_image.dtype, val_label.shape))
print("Test info")
print("Image : {}({}), Label : {}".format(test_image.shape, test_image.dtype, test_label.shape))

train_class_count = label_count(train_label, CIFARNCLASS)
val_class_count = label_count(val_label, CIFARNCLASS)
test_class_count = label_count(test_label, CIFARNCLASS)

print("Train class mean : {}, std : {}".format(np.mean(train_class_count), np.std(train_class_count)))
print("Val class mean : {}, std : {}".format(np.mean(val_class_count), np.std(val_class_count)))
print("Test class mean : {}, std : {}".format(np.mean(test_class_count), np.std(test_class_count)))

create_muldir(CIFARPROCESSED)
write_npy(train_image, CIFARPROCESSED+'train_image.npy') 
write_npy(train_label, CIFARPROCESSED+'train_label.npy') 
write_npy(val_image, CIFARPROCESSED+'val_image.npy') 
write_npy(val_label, CIFARPROCESSED+'val_label.npy') 
write_npy(test_image, CIFARPROCESSED+'test_image.npy') 
write_npy(test_label, CIFARPROCESSED+'test_label.npy') 
