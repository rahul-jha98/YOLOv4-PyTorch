# YOLOv4-PyTorch

> **Minimal PyTorch implementation of YOLOv4**

### 1. Setting Up Environment

To train the model you would need a python environemnt with all the dependencies mentioned in requirements.txt installed. 
To install the dependencies run :

```
$ pip install -r requirements.txt
```

### 2. Preparing Data

**Folder Structure**

To prepare your dataset for training using our implementation you will have to prepare two folders one for images and one for labels. For any image path in image folder a corresponding label file (same name but .txt extension) is searched in the labels folder. If the label file is found the image and labels will be used in the training process. 

Here is an example of the directory structure for the train folder. 

![](../image_assets/directory_structure.png)

**File Structure for each label file**

For each image in image folder its label file is a txt file which will contain the details of all the bounding boxes in the image. The format of the txt file is as follows 

```
x1, y1, x2, y2, id
x1, y1, x2, y1, id
...
```
If there are N objects in the image the txt will contain N lines each containing the following values:

* x1, y1: coordinate of the upper left corner
* x2, y2: coordinate of the lower left corner
* id: category id of the object




### 3. Training

One you have prepared your data and set the train path and validation path in the in the `cfg.py` file you can use `train.py` to begin training your model.

```
$ python train.py -g [GPU_ID]
```

Additional arguments could be provided to change some of the config parameters. 

The config parameters and their default values are mentioned below.

Argument|Default|Description
---|---|---
`-l` or `--load`|None|Path of the pretrained weight file (.pth) to load before training.
`--classes`|80|Number of dataset classes
`-r` or `--learning-rate`|0.001|Value of learning rate to use during training
`-g` or `--gpu`| -1 | Parameter for CUDA_VISIBLE_DEVICES (default value -1 will use CPU for training)
`--optimizer`| `adam` | Choose one optimizer from `adam` or `sgd`
