# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import cv2
from models import *
import torch
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    image = cv2.resize(image,(128,128))
    if image is None:
        return None

    # horizontal = cv2.flip(image,1,dst=None)
    # image = horizontal

    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    pbar = tqdm(len(test_list))
    for i, img_path in enumerate(test_list):
        pbar.update(1)
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


if __name__ == '__main__':

    batch_size = 32
    data_dir = 'D:/pyCharm/opensource-face/dataset/Test_Data/'
    name_list = [name for name in os.listdir(data_dir)]
    img_paths = [data_dir+name for name in os.listdir(data_dir)]
    print('Images number:', len(img_paths))

    model = resnet_face18(False)
    state_dict = torch.load('resnet18.pth')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v     
    model.load_state_dict(new_state_dict)
    model.to(torch.device("cuda"))
    model.eval()
    
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))

    fe_dict = get_feature_dict(name_list, features)
    print('Output number:', len(fe_dict))
    sio.savemat('face_embedding_test.mat', fe_dict)
