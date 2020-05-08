import torch
import scipy.io
import matplotlib.pylab as plt
import os
import cv2
import tqdm
from sklearn.model_selection import train_test_split
import glob
from cls_hrnet import HighResolutionNet, get_cls_net
import json
from torchvision import transforms
from PIL import Image
import numpy as np

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((800, 544)), transforms.ToTensor()])

    folder = 'data'
    output_folder = folder + '/annotations/pixel-level/'
    input_folder = folder + '/photos/'

    label_list = scipy.io.loadmat(folder + '/' + 'label_list.mat')

    inputs = [transform(Image.open(x)).unsqueeze(0) for x in sorted(glob.glob(f'{input_folder}*.jpg'))[:2]] # 1004
    labels = [scipy.io.loadmat(x)['groundtruth'] for x in sorted(glob.glob(f'{output_folder}*.mat'))[:2]]

    inputs = torch.cat(inputs)

    # X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, shuffle=True)
    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

    with open('params.json', 'r') as file:
        params = json.load(file)

    model = get_cls_net(params)

    output1 = model(inputs)
    print(output1.shape)