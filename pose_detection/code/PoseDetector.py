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
from matplotlib import cm
from pose_detection.code import hopenet, hopelessnet, utils


class PoseDetector:
    cudnn.enabled = True
    batch_size = 1
    gpu = None
    
    out_dir = 'output'
    model = None
    transformations = None

    def __init__(self, arch='ResNet50', snapshot_path='./pose_detection/models/hopenet_snapshot_a1.pkl'):
        if arch == 'ResNet18':
            self.model = hopenet.Hopenet(
                torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
        elif arch == 'ResNet34':
            self.model = hopenet.Hopenet(
                torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], 66)
        elif arch == 'ResNet101':
            self.model = hopenet.Hopenet(
                torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
        elif arch == 'ResNet152':
            self.model = hopenet.Hopenet(
                torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], 66)
        elif arch == 'Squeezenet_1_0':
            self.model = hopelessnet.Hopeless_Squeezenet(arch, 66)
        elif arch == 'Squeezenet_1_1':
            self.model = hopelessnet.Hopeless_Squeezenet(arch, 66)
        elif arch == 'MobileNetV2':
            self.model = hopelessnet.Hopeless_MobileNetV2(66, 1.0)
        else:
            if arch != 'ResNet50':
                print('Invalid value for architecture is passed! '
                    'The default value of ResNet50 will be used instead!')
            self.model = hopenet.Hopenet(
                torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        # print('Loading snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path,  map_location={'cuda:0': 'cpu'})
        self.model.load_state_dict(saved_state_dict)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        self.transformations = transforms.Compose([transforms.Scale(224),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # self.model.cuda(self.gpu)

        print('Ready to test network.')
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    def detect(self, image_path, isFile, show=False):
        idx_tensor = [idx for idx in range(66)]
        # idx_tensor = torch.FloatTensor(idx_tensor).cuda(self.gpu)
        idx_tensor = torch.FloatTensor(idx_tensor)

        if isFile:
            print("opening image:" + image_path)
            image = cv2.imread(image_path)
        else:
            image = image_path
            image *= 255.0

        height, width, channels = image.shape
        img = Image.fromarray(image.astype('uint8'))

        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        # img = Variable(img).cuda(self.gpu)
        img = Variable(img)

        yaw, pitch, roll = self.model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        # print('output:  %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
        output = utils.draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx = width/2, tdy= height/2, size = height/2)

        yaw_predicted = float(yaw_predicted.cpu().numpy())
        pitch_predicted = float(pitch_predicted.cpu().numpy())
        roll_predicted = float(roll_predicted.cpu().numpy())

        if show:
            cv2.imwrite(str(roll_predicted)+".jpg", output)
            # cv2.imshow('window', output)

            cv2.waitKey(-1)
        return yaw_predicted, pitch_predicted, roll_predicted

