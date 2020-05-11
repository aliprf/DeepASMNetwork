import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, hopelessnet, utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', 
        dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--snapshot', 
        dest='snapshot', help='Path of model snapshot.', default='', type=str)
    parser.add_argument('--image', 
        dest='image_path', help='Path of video')
    parser.add_argument('--bboxes', 
        dest='bboxes', help='Bounding box annotations of frames')
    parser.add_argument('--output_string', 
        dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', 
        dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', 
        dest='fps', help='Frames per second of source video', 
        type=float, default=30.)
    parser.add_argument('--arch', 
        dest='arch', 
        help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], '
            'ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output'
    image_path = args.image_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Base network structure
    if args.arch == 'ResNet18':
        model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
    elif args.arch == 'ResNet34':
        model = hopenet.Hopenet(
            torchvision.models.resnet.BasicBlock, [3,4,6,3], 66)
    elif args.arch == 'ResNet101':
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
    elif args.arch == 'ResNet152':
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
    elif args.arch == 'Squeezenet_1_0':
        model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
    elif args.arch == 'Squeezenet_1_1':
        model = hopelessnet.Hopeless_Squeezenet(args.arch, 66)
    elif args.arch == 'MobileNetV2':
        model = hopelessnet.Hopeless_MobileNetV2(66, 1.0)
    else:
        if args.arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    print("opening image:" + image_path)
    image = cv2.imread(image_path)
    height, width, channels = image.shape
   
    cv_img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

 
    img = Image.fromarray(cv_img)

    # Transform
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda(gpu)

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

    print('output:  %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
    output= utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx = width/2, tdy= height/2, size = height/2)
    cv2.imshow('window', output)
    cv2.waitKey(-1)

