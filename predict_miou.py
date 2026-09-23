from utils.utils import cvtColor, preprocess_input, resize_image
import time
import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from utils.utils_metrics import f_score, IOUMetric
from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image
import cv2
# import os


class Unet(object):
    _defaults = {

        "model_path": 'logs\ep100-loss0.233-val_loss0.253.pth',
        "num_classes": 2,
        "backbone": "vgg",
        "input_shape": [512, 512],
        "mix_type": 0,
        "cuda": True,
    }
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
    def generate(self):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    def detect_image(self, image,label):
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)
            pr = F.softmax(pr.permute(0,2, 3, 1), dim=-1).cpu().numpy()
            # pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
            #      int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
            MIOU_Metrics = IOUMetric(2)
            # for batch in range(len(imgs)):
            # print(pr.shape)
            label = np.array(Image.open(label))
            label = label / 255.0 #label type: float64
            print("label type:",label.dtype)
            label = np.expand_dims(label,0)
            print(pr.shape,label.shape)
            meitrics = MIOU_Metrics.evaluate(pr, label)
            print("miou:",meitrics[0])

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
            # ------------------------------------------------#
            #   将新图与原图及进行混合
            # ------------------------------------------------#
            image = Image.blend(old_img, image, 0.7)

if __name__ == "__main__":

    unet = Unet()

    mode = "predict"

    label_path = "D:/Algorithm/DeepLearning/语义分割/Unet_chuan/Breast_Datasets/Labels/02148279_b_T2W10.png"
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(dir_origin_path+img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image,label_path)
                # r_image.show()


    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
