# Image Segmentation Keras : Implementation of Segnet, FCN, UNet, PSPNet and other models in Keras.

[![PyPI version](https://badge.fury.io/py/keras-segmentation.svg)](https://badge.fury.io/py/keras-segmentation)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/divamgupta)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)


Implementation of various Deep Image Segmentation models in keras. 

Link to the full blog post with tutorial : https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html

<p align="center">
  <img src="https://raw.githubusercontent.com/sunshineatnoon/Paper-Collection/master/images/FCN1.png" width="50%" >
</p>

## Updates to original 
- Added accuracy and loss plot functions to 'keras_segementation/metrics.py'
- Added an 'output_directory' parameter in 'keras_segementation/train.py' to output accuracy and loss plots to a specificed directory

## Using the python module (updated)

You can import keras_segmentation in  your python script and use the API

```python
import numpy as np 
import keras_segmentation
import os
import time
import sys

start_time = time.time()

#paths
input_dir = '/data/' #directory with all the input images and annotations
img_train_dir = os.path.join(input_dir, 'train_images/')
ann_train_dir = os.path.join(input_dir, 'train_annotations/')
img_val_dir = os.path.join(input_dir, 'val_images/')
ann_val_dir = os.path.join(input_dir, 'val_annotations/')
img_test_dir = os.path.join(input_dir, 'test_images/')
ann_test_dir = os.path.join(input_dir, 'test_annotations/')
out_dir = '/output/' #directory to store all the image outputs

num_classes = 3     #number of classes in the images
model = keras_segmentation.models.unet.vgg_unet(n_classes=num_classes,  input_height=416, input_width=608  ) #for vgg16 encoder and unet network

model.train( 
    train_images =  img_train_dir,
    train_annotations = ann_train_dir,
    validate=True,
    val_images = img_val_dir,
    val_annotations = ann_val_dir,
    epochs = 5,
    optimizer_name = 'SGD',
    output_directory = out_dir
    )
    
#Make a list of the test images and annotations
test_img = []
test_ann = []
    
for img_path in os.listdir(img_test_dir):
    test_img.append(os.path.join(img_test_dir, img_path))
        
for img_path in os.listdir(ann_test_dir):
    test_ann.append(os.path.join(ann_test_dir, img_path))
    
#evaluate the test images (which prints the IoUs)
model.evaluate_segmentation(inp_images = test_img, annotations = test_ann)

#predict the segmentations in test images, images saved in out_dir
out = model.predict_multiple(inp_dir = img_test_dir, out_dir = out_dir)

print("Total runtime: {}".format(time.time() - start_time))
)
```
## Other Repositories 
- [Attention based Language Translation in Keras](https://github.com/divamgupta/attention-translation-keras)
- [Ladder Network in Keras](https://github.com/divamgupta/ladder_network_keras)  model achives 98% test accuracy on MNIST with just 100 labeled examples

### Contributors 

Divam Gupta : https://divamgupta.com


## Models 

Following models are supported:

| model_name       | Base Model        | Segmentation Model |
|------------------|-------------------|--------------------|
| fcn_8            | Vanilla CNN       | FCN8               |
| fcn_32           | Vanilla CNN       | FCN8               |
| fcn_8_vgg        | VGG 16            | FCN8               |
| fcn_32_vgg       | VGG 16            | FCN32              |
| fcn_8_resnet50   | Resnet-50         | FCN32              |
| fcn_32_resnet50  | Resnet-50         | FCN32              |
| fcn_8_mobilenet  | MobileNet         | FCN32              |
| fcn_32_mobilenet | MobileNet         | FCN32              |
| pspnet           | Vanilla CNN       | PSPNet             |
| vgg_pspnet       | VGG 16            | PSPNet             |
| resnet50_pspnet  | Resnet-50         | PSPNet             |
| unet_mini        | Vanilla Mini CNN  | U-Net              |
| unet             | Vanilla CNN       | U-Net              |
| vgg_unet         | VGG 16            | U-Net              |
| resnet50_unet    | Resnet-50         | U-Net              |
| mobilenet_unet   | MobileNet         | U-Net              |
| segnet           | Vanilla CNN       | Segnet             |
| vgg_segnet       | VGG 16            | Segnet             |
| resnet50_segnet  | Resnet-50         | Segnet             |
| mobilenet_segnet | MobileNet         | Segnet             |


Example results for the pre-trained models provided :

Input Image            |  Output Segmentation Image 
:-------------------------:|:-------------------------:
![](sample_images/1_input.jpg)  |  ![](sample_images/1_output.png)
![](sample_images/3_input.jpg)  |  ![](sample_images/3_output.png)


## Getting Started

### Prerequisites

* Keras 2.0
* opencv for python
* Theano / Tensorflow / CNTK 

```shell
sudo apt-get install python-opencv
sudo pip install --upgrade keras
```

### Installing

Install the module

```shell
pip install keras-segmentation
```

### or

```shell
git clone https://github.com/divamgupta/image-segmentation-keras
cd image-segmentation-keras
python setup.py install
```
pip install will be available soon!


## Pre-trained models:
```python
import keras_segmentation

model = keras_segmentation.pretrained.pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

model = keras_segmentation.pretrained.pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

model = keras_segmentation.pretrained.pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="input_image.jpg",
    out_fname="out.png"
)

```


### Preparing the data for training

You need to make two folders

*  Images Folder - For all the training images 
* Annotations Folder - For the corresponding ground truth segmentation images

The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same. 

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

Example code to generate annotation images :

```python
import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
```

Only use bmp or png format for the annotation images.

## Download the sample prepared dataset

Download and extract the following:

https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing

You will get a folder named dataset1/ 


## Using the python module

You can import keras_segmentation in  your python script and use the API

```python
import keras_segmentation

model = keras_segmentation.models.unet.vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )

model.train( 
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="/tmp/out.png"
)


import matplotlib.pyplot as plt
plt.imshow(out)

```


## Usage via command line 
You can also use the tool just using command line

### Visualizing the prepared data

You can also visualize your prepared annotations for verification of the prepared data.


```shell
python -m keras_segmentation verify_dataset \
 --images_path="dataset1/images_prepped_train/" \
 --segs_path="dataset1/annotations_prepped_train/"  \
 --n_classes=50
```

```shell
python -m keras_segmentation visualize_dataset \
 --images_path="dataset1/images_prepped_train/" \
 --segs_path="dataset1/annotations_prepped_train/"  \
 --n_classes=50
```



### Training the Model

To train the model run the following command:

```shell
python -m keras_segmentation train \
 --checkpoints_path="path_to_checkpoints" \
 --train_images="dataset1/images_prepped_train/" \
 --train_annotations="dataset1/annotations_prepped_train/" \
 --val_images="dataset1/images_prepped_test/" \
 --val_annotations="dataset1/annotations_prepped_test/" \
 --n_classes=50 \
 --input_height=320 \
 --input_width=640 \
 --model_name="vgg_unet"
```

Choose model_name from the table above



### Getting the predictions

To get the predictions of a trained model

```shell
python -m keras_segmentation predict \
 --checkpoints_path="path_to_checkpoints" \
 --input_path="dataset1/images_prepped_test/" \
 --output_path="path_to_predictions"

```



## Fine-tuning from existing segmentation model

The following example shows how to fine-tune a model with 10 classes .

```python
import keras_segmentation
from keras_segmentation.models.model_utils import transfer_weights


pretrained_model = keras_segmentation.pretrained.pspnet_50_ADE_20K() 

new_model = keras_segmentation.models.pspnet.pspnet_50( n_classes=51 )

transfer_weights( pretrained_model , new_model  ) # transfer weights from pre-trained model to your model

new_model.train( 
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "/tmp/vgg_unet_1" , epochs=5
)


```

