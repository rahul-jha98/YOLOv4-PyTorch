import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.train_path = os.path.join(_BASE_DIR, 'data', 'train')
Cfg.validation_path = os.path.join(_BASE_DIR, 'data', 'valid')
Cfg.width = 608
Cfg.height = 608

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