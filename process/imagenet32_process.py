import sys
sys.path.append('../configs/')
sys.path.append('../utils/')

# ../configs
from path import IMAGENET32PATH, IMAGENET32PROCESSED

# ../utils
from reader import read_pkl
from writer import write_npy, create_muldir

from sklearn.model_selection import train_test_split
import numpy as np

train_img = list()
train_label = list()
image_mean = None

for idx in range(1,10):
    content = read_pkl(IMAGENET32PATH+'train_data_batch_%d'%idx)
    if image_mean is None: image_mean = content['mean']
    else: assert np.sum(image_mean)==np.sum(content['mean']), "pixel_mean value should be same"

    nimg = len(content['data'])
    nlabel = len(content['labels'])
    assert nimg==nlabel, "image and label should be same "
    train_img.append(content['data'])
    train_label.append(content['labels'])
    
train_img = np.concatenate(train_img, axis=0)
train_label = np.concatenate(train_label, axis=0)-1

def imagenet_image_processor(img_, mean):
    '''
    Args:
        img_ - Numpy 2D [ndata, 32*32*3]
        mean - [32*32,3]
    '''
    tmp = (img_-mean)/np.float32(255)
    tmp = tmp.astype(np.float32)
    tmp = np.transpose(tmp.reshape([-1,3,32,32]), [0, 2, 3, 1])
    return tmp
train_img = imagenet_image_processor(train_img, image_mean)

content = read_pkl(IMAGENET32PATH+'val_data') # val data used as test data
test_img = imagenet_image_processor(content['data'], image_mean)
test_label = np.array(content['labels'])-1

content = read_pkl(IMAGENET32PATH+'train_data_batch_10')
val_img = imagenet_image_processor(content['data'], image_mean)
val_label = np.array(content['labels'])-1

val_label_set = list()
for i in range(1000): val_label_set.append(list())

for i in range(len(val_label)): val_label_set[val_label[i]].append(i)

new_val_idx = list()
for i in range(1000): new_val_idx.extend(val_label_set[i][:50])

new_val_idx = np.array(new_val_idx) 

new_val_img = val_img[new_val_idx]
new_val_label = val_label[new_val_idx]

print("Train Image : {}({}, max={}, min={}), Label : {}".format(train_img.shape, train_img.dtype, np.max(train_img), np.min(train_img), train_label.shape))
print("Val Image : {}({}, max={}, min={}), Label : {}".format(new_val_img.shape, new_val_img.dtype, np.max(new_val_img), np.min(new_val_img), new_val_label.shape))
print("Test Image : {}({}, max={}, min={}), Label : {}".format(test_img.shape, test_img.dtype, np.max(test_img), np.min(test_img), test_label.shape))

create_muldir(IMAGENET32PROCESSED)
write_npy(train_img, IMAGENET32PROCESSED+'train_img.npy') 
write_npy(train_label, IMAGENET32PROCESSED+'train_label.npy') 
write_npy(new_val_img, IMAGENET32PROCESSED+'val_img.npy') 
write_npy(new_val_label, IMAGENET32PROCESSED+'val_label.npy') 
write_npy(test_img, IMAGENET32PROCESSED+'test_img.npy') 
write_npy(test_label, IMAGENET32PROCESSED+'test_label.npy') 
