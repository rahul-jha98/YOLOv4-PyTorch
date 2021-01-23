import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

# Config file to be used
Cfg.use_darknet_cfg = False
Cfg.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')

'''
Default path is set as 
data/
-- train/
------ Images/          - Folder for train images
------ Annotations/     - Folder for train annotations
-- valid/
------ Images/          - Folder for validation images
------ Annotations/     - Folder for validation annotations
'''
Cfg.train_path = os.path.join(_BASE_DIR, 'data', 'train')
Cfg.train_img_dir = os.path.join(Cfg.train_path, 'Images')
Cfg.train_label_dir = os.path.join(Cfg.train_path, 'Annotations')

Cfg.use_validation = False
Cfg.validation_path = os.path.join(_BASE_DIR, 'data', 'valid')
Cfg.valid_img_dir = os.path.join(Cfg.train_path, 'Images')
Cfg.valid_label_dir = os.path.join(Cfg.train_path, 'Annotations')


# Training loop parameters
Cfg.TRAIN_EPOCHS = 300
Cfg.TRAIN_OPTIMIZER = 'adam'

Cfg.batch = 64
Cfg.subdivisions = 16
Cfg.width = 608
Cfg.height = 608
Cfg.channels = 3
Cfg.momentum = 0.949
Cfg.decay = 0.0005

Cfg.learning_rate = 0.00261
Cfg.burn_in = 1000
Cfg.max_batches = 500500
Cfg.steps = [400000, 450000]
Cfg.policy = Cfg.steps
Cfg.scales = .1, .1

# Parameters for transforms
Cfg.angle = 0
Cfg.saturation = 1.5
Cfg.exposure = 1.5
Cfg.hue = .1

Cfg.boxes = 60
Cfg.mosaic = 1
Cfg.letter_box = 0
Cfg.jitter = 0.2
Cfg.classes = 80
Cfg.track = 0
Cfg.w = Cfg.width
Cfg.h = Cfg.height
Cfg.flip = 1
Cfg.blur = 0
Cfg.gaussian = 0
Cfg.mixup = 0


Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

Cfg.keep_checkpoint_max = 10