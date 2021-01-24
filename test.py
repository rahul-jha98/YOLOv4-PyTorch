import sys
import cv2
import os
import numpy as np
import torch

from models import Yolov4
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
    
import argparse

def get_args():
    parser = argparse.ArgumentParser('Test the trained model on your images')
    parser.add_argument('-i', '--input', type=str,
                        help='path of your image file or image directory', dest='input')
    parser.add_argument('--weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-n', '--num_classes', type=int, default=80,
                        help='number of classes in dataset', dest='num_classes')
    parser.add_argument('--width', type=int, default=608,
                        help='width of image to be used in inference', dest='width')
    parser.add_argument('--height', type=int, default=608,
                        help='height of image to be used in inference', dest='height')
    parser.add_argument('--namesfile', type=str, default='data/coco.names',
                        help='file with all the names of classes', dest='namesfile')
    parser.add_argument('-o', '--output', type=str, default='output.jpg',
                        help='name of the output file (or output directory if input is directory)', dest='output')
    args = parser.parse_args()

    return args
  

if __name__ == "__main__":
    args = get_args()
    
    model = Yolov4(yolov4conv137weight=None, n_classes=args.num_classes, inference=True)

    pretrained_dict = torch.load(args.weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    
    use_cuda = True
    if use_cuda:
        model.cuda()
    
    ## Warming up the model
    do_detect(model, np.zeros((args.height,args.width,3), np.uint8), .4, .6, use_cuda)
    
    if os.path.isdir(args.input):
        
        
        for parent, _, files_list in os.walk(args.input):
            parent_path = os.path.relpath(parent, args.input) 
            if parent_path.startswith('./'):
                parent_path = parent_path[2:]
                
            for file in files_list:
                img_path = os.path.join(parent, file)
                rel_img_path = os.path.join(parent_path, file)
                
                img = cv2.imread(img_path)
                
                sized = cv2.resize(img, (args.width, args.height))
                sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
                
                boxes = do_detect(model, sized, 0.4, 0.6, use_cuda, False)
                class_names = load_class_names(args.namesfile)
                
                output_path = os.path.join(args.output, rel_img_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plot_boxes_cv2(img, boxes[0], output_path, class_names)
                
    else:
        img = cv2.imread(args.input)
    
        # Inference input size is 608*608 does not mean training size is the same
        # Training size could be 608*608 or even other sizes
        # Optional inference sizes:
        #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
        #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
        sized = cv2.resize(img, (args.width, args.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
        class_names = load_class_names(args.namesfile)
        
        try:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        except FileNotFoundError:
            pass
        
        plot_boxes_cv2(img, boxes[0], args.output, class_names)