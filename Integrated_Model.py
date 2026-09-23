import os
import time
import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pip._internal.cli.cmdoptions import pre
from torch import nn
import matplotlib.pyplot as plt

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, cvtColor_new

from nets.unet import Unet as unet
from nets.SegNet_VGG16 import SegNet
from nets.VGG16_Deeplab_v3plus import VGG16_DeepLabv3_plus
class Unet(object):
    def __init__(self, num_classes=2, backbone='vgg', model_path='logs'):
        self.num_classes = num_classes
        self.backbone = backbone
        self.cuda = True
        # 获得模型
        self.model_path = model_path
        self.generate_model()

    def generate_model(self):
        self.net = unet(num_classes=self.num_classes, backbone=self.backbone)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('model parameters {} has been loaded into Network'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # def predict_image(self,image_path):
    #     pr = self.Net_output(image_path)
    #     pr = pr.argmax(axis=-1) #size: 512*512
    #     # plt.imshow(pr)
    #     # plt.show()
    #     pr = pr.reshape([-1]) #size: 512*512
    #     # print("len()!=0:",len(pr[pr!=0]))
    #     return pr
    #
    # def Net_output(self, image_path):
    #     # image = Image.open(image_path)
    #     image = Image.fromarray(np.load(image_path))
    #     image = cvtColor(image)
    #     # image = cvtColor_new(image) #image.shape = (3, 256, 256)
    #     # 增加batchsize维度
    #     # print("image.shape = {}".format(image.shape))
    #     image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)
    #     # image_data = np.expand_dims(image.astype(np.float32),0) #image_data.shape = (1, 3, 256, 256)
    #     # print("image_data.shape = {}".format(image_data.shape))
    #     with torch.no_grad():  # 这样会没有反向传播的属性，降低显存开销
    #         images = torch.from_numpy(image_data)  # torch.Size([1, 3, 512, 512])
    #         if self.cuda:
    #             images = images.cuda()
    #
    #         # 图片传入网络
    #         pr = self.net(images)[0]  # size: (2, 512, 512)
    #         # print(pr)
    #
    #         pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()  # size: (512, 512, 2)
    #
    #     return pr

def Dice_score(label, inputs, num_classes):#Label size: (512, 512)
    smooth = 1e-5
    integrate_input = np.zeros(inputs[0].shape, dtype=np.float64)
    for input in inputs:
        integrate_input += input
    # integrate_output.shape: (512, 512, 2)
    integrate_output = integrate_input / len(inputs)
    label = np.array(label)
    modify_label = np.zeros_like(label)
    modify_label[label > 127.5] = 1
    seg_label = modify_label
    # seg_label.shape: (512, 512, 2)
    seg_label = np.eye(num_classes)[seg_label.reshape([-1])]
    seg_label = seg_label.reshape((label.shape[0],label.shape[1], num_classes))

    # integrate_output = np.transpose(integrate_output,(2,0,1))
    # seg_label = np.transpose(seg_label,(2,0,1))

    seg_label = np.expand_dims(seg_label, axis=0)
    integrate_output = np.expand_dims(integrate_output, axis=0)

    seg_label = torch.tensor(seg_label) #seg_label.size: torch.Size([1, 512, 512, 2])
    integrate_output = torch.tensor(integrate_output) #integrate_output.size: torch.Size([1, 512, 512, 2])

    n, h, w, c = integrate_output.size()
    nt, ht, wt, ct = seg_label.size()

    temp_input = integrate_output.view(n, -1, c) #temp_input.size: torch.Size([1, 262144, 2])
    temp_label = seg_label.view(nt, -1, ct) #temp_label.size: torch.Size([1, 262144, 2])

    # print("temp_input.size",temp_input.size())
    # print("temp_label.size",temp_label.size())

    tp = torch.sum(temp_label*temp_input, axis=[0, 1])
    fp = torch.sum(temp_input, axis=[0, 1]) - tp
    fn = torch.sum(temp_label, axis=[0, 1]) - tp

    score = (2 * tp + smooth) / (2 * tp + 2 * fn + fp + smooth)
    return torch.mean(score)

def Integrate_predict(pre_modal, label, num_classes):
    smooth = 1e-5
    label = label.convert('L')
    # plt.imshow(label)
    # plt.show()
    input_shape = label.size
    label = np.array(label)
    label = label.reshape([-1]).astype(np.int64)

    Integrate_pre = []

    for i in range(len(label)):
        temp_ele = []
        for _, pre in pre_modal.items():
            temp_ele.append(pre[i])
        # temp_ele.append(pre1[i])
        # temp_ele.append(pre2[i])
        # temp_ele.append(pre3[i])
        # 统计出现次数最多的元素
        Integrate_pre_ele = max(temp_ele, key=temp_ele.count)
        # if 1 in temp_ele:
        #     Integrate_pre_ele = 1
        # else:
        #     Integrate_pre_ele = 0
        Integrate_pre.append(Integrate_pre_ele)
    Integrate_pre = np.array(Integrate_pre)
    return hist_operation(num_classes, label, Integrate_pre), Integrate_pre.reshape(input_shape)

def hist_operation(num_classes, label, pre, smooth=1e-5):
    # label = np.array(label)
    if np.shape(label) != np.shape(pre):
        label = label.convert('L')
        label = np.array(label)
        label = label.reshape([-1]).astype(np.int64)
    mask = (label >= 0) & (label < num_classes)
    # print("label and pre shape:{},{}".format(np.shape(label),np.shape(pre)))
    hist = np.bincount(
        num_classes * label[mask].astype(int) +
        pre[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    # print('hist:',hist)
    TN, FP, FN, TP = hist[0][0], hist[0][1], hist[1][0], hist[1][1]
    iou = TP / (TP + FP + FN + smooth)
    Sensitivity = TP / (TP + FN + smooth)
    PPV = TP / (TP + FP + smooth)
    Dice = (2 * TP) / (FP + 2 * TP + FN + smooth)
    OR = FP / (FP + FN + TP + smooth)
    UR = FN / (FP + FN + TP + smooth)
    MA = TP / (TP + FP + TN + FN + smooth)
    # Pre = TP / (TP + FP + smooth)
    return iou, Sensitivity, PPV, Dice,OR, UR

def predict_image(image_path, model):
    pr = Net_output(image_path, model)
    pr = pr.argmax(axis=-1) #size: 512*512
    # plt.imshow(pr)
    # plt.show()
    pr = pr.reshape([-1]) #size: 512*512
    # print("len()!=0:",len(pr[pr!=0]))
    return pr

def Net_output(image_path, model):
    # image = Image.open(image_path)
    image = Image.fromarray(np.load(image_path))
    image = cvtColor(image)
    # image = cvtColor_new(image) #image.shape = (3, 256, 256)
    # 增加batchsize维度
    # print("image.shape = {}".format(image.shape))
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)
    # image_data = np.expand_dims(image.astype(np.float32),0) #image_data.shape = (1, 3, 256, 256)
    # print("image_data.shape = {}".format(image_data.shape))
    with torch.no_grad():  # 这样会没有反向传播的属性，降低显存开销
        images = torch.from_numpy(image_data)  # torch.Size([1, 3, 512, 512])

        images = images.cuda()

        # 图片传入网络
        pr = model(images)[0]  # size: (2, 512, 512)
        # print(pr)

        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()  # size: (512, 512, 2)

    return pr

if __name__ == '__main__':
    # modals = ['T1W', 'T2W', 'T1WFLAIR', 'T2WFLAIR',
               # 'PDMapping', 'T1Mapping']
    # modals = ['T1W', 'T2WFLAIR','PDMapping', 'T1Mapping', 'T2Mapping']
    modals = ['T1W', 'T1WFLAIR', 'T2WFLAIR']
    modal_len = len(modals)
    model_path = {'T1W': r'logs\Unet\T1W\ep200-loss0.010-val_loss0.000-moiu0.646-dice0.883.pth',
                  'T2W': r'logs\Unet\T2W\ep187-loss0.007-val_loss0.000-moiu0.586-dice0.859.pth',
                  'T1WFLAIR': r'logs\Unet\T1WFLAIR\ep196-loss0.003-val_loss0.000-moiu0.652-dice0.885.pth',
                  'T2WFLAIR': r'logs\Unet\T2WFLAIR\ep191-loss0.002-val_loss0.000-moiu0.644-dice0.882.pth',
                  'STIR': r'logs\Unet\STIR\ep189-loss0.008-val_loss0.000-moiu0.427-dice0.782.pth',
                  'PDW': r'logs\Unet\PDW\ep191-loss0.003-val_loss0.000-moiu0.562-dice0.845.pth',
                  'PDMapping': r'logs\Unet\PDMapping\ep187-loss0.012-val_loss0.000-moiu0.539-dice0.835.pth',
                  'T1Mapping': r'logs\Unet\T1Mapping\ep198-loss0.004-val_loss0.000-moiu0.575-dice0.852.pth',
                  'T2Mapping': r'logs\Unet\T2Mapping\ep196-loss0.006-val_loss0.000-moiu0.485-dice0.812.pth'
                  }
    # model_path = {
    #               # 'T2W': 'logs\SegNet\T2W\ep187-loss0.007-val_loss0.000-moiu0.586-dice0.859.pth',
    #               # 'T1WFLAIR': 'logs\SegNet\T1WFLAIR\ep196-loss0.003-val_loss0.000-moiu0.652-dice0.885.pth',
    #               # 'T2WFLAIR': 'logs\SegNet\T2WFLAIR\ep191-loss0.002-val_loss0.000-moiu0.644-dice0.882.pth',
    #               # 'STIR': 'logs\SegNet\STIR\ep189-loss0.008-val_loss0.000-moiu0.427-dice0.782.pth',
    #               # 'PDW': 'logs\SegNet\PDW\ep191-loss0.003-val_loss0.000-moiu0.562-dice0.845.pth',
    #               'PDMapping': r'logs\SegNet\PDMapping\ep030-loss0.043-val_loss0.000-moiu0.472-dice0.803.pth',
    #               'T1Mapping': r'logs\SegNet\T1Mapping\ep011-loss0.086-val_loss0.001-moiu0.485-dice0.811.pth',
    #               'T2Mapping': r'logs\SegNet\T2Mapping\ep008-loss0.132-val_loss0.001-moiu0.370-dice0.750.pth'
    #               }

    # model_path = {'T1W': 'logs\DeepLabV3+\T1W\ep121-loss0.093-val_loss0.001-moiu0.555-dice0.837.pth',
    #               'T2W': 'logs\DeepLabV3+\T2W\ep147-loss0.118-val_loss0.001-moiu0.440-dice0.783.pth',
    #               'T1WFLAIR': 'logs\DeepLabV3+\T1WFLAIR\ep182-loss0.071-val_loss0.001-moiu0.627-dice0.872.pth',
    #               'T2WFLAIR': 'logs\DeepLabV3+\T2WFLAIR\ep164-loss0.088-val_loss0.001-moiu0.531-dice0.828.pth',
    #               # 'STIR': 'logs\DeepLabV3+\STIR\ep189-loss0.008-val_loss0.000-moiu0.427-dice0.782.pth',
    #               'PDW': 'logs\DeepLabV3+\PDW\ep191-loss0.003-val_loss0.000-moiu0.562-dice0.845.pth',
    #               'PDMapping': r'logs\DeepLabV3+\PDMapping\ep199-loss0.092-val_loss0.001-moiu0.536-dice0.831.pth',
    #               'T1Mapping': r'logs\DeepLabV3+\T1Mapping\ep170-loss0.035-val_loss0.000-moiu0.512-dice0.820.pth',
    #               'T2Mapping': r'logs\DeepLabV3+\T2Mapping\ep199-loss0.057-val_loss0.000-moiu0.344-dice0.733.pth'
    #               }
    backbone = 'vgg'
    num_classes = 2
    model_dict = dict()
    for modal in modals:
        model_dict[modal] = unet(num_classes)
        # model_dict[modal] = VGG16_DeepLabv3_plus(n_classes=num_classes)
        # model_dict[modal] = SegNet(class_num=num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict[modal].load_state_dict(torch.load(model_path[modal], map_location=device))
        model_dict[modal] = model_dict[modal].eval()
        print('model parameters {} has been loaded into Network'.format(model_path[modal]))
        model_dict[modal] = nn.DataParallel(model_dict[modal])
        model_dict[modal] = model_dict[modal].cuda()


    val_txt_path = {'T1W': 'Breast_Datasets\T1W\ImageSets\Segmentation\\trainval.txt',
                    'T1WFLAIR': 'Breast_Datasets\T1WFLAIR\ImageSets\Segmentation\\trainval.txt',
                    'T2WFLAIR': 'Breast_Datasets\T2WFLAIR\ImageSets\Segmentation\\trainval.txt',
                    'T2W': 'Breast_Datasets\T2W\ImageSets\Segmentation\\trainval.txt',
                    'STIR': 'Breast_Datasets\STIR\ImageSets\Segmentation\\trainval.txt',
                    'PDW': 'Breast_Datasets\PDW\ImageSets\Segmentation\\trainval.txt',
                    'PDMapping': 'Breast_Datasets\PDMapping\ImageSets\Segmentation\\trainval.txt',
                    'T1Mapping': 'Breast_Datasets\T1Mapping\ImageSets\Segmentation\\trainval.txt',
                    'T2Mapping': 'Breast_Datasets\T2Mapping\ImageSets\Segmentation\\trainval.txt'
                    }

    val_img_list = dict()

    for modal, data_path in val_txt_path.items():
        with open(data_path) as f:
            lines = f.readlines()
            lines = [line.split('\n')[0] for line in lines]
            lines.sort()
            val_img_list[modal] = lines
    val_set_len = len(val_img_list['T1W'])

    data_dir = os.getcwd() + '\Breast_Datasets'

    Miou, Sensitivity, PPV, Dice, UR, OR = 0, 0, 0, 0, 0, 0
    # integrate_metrics = {'Miou': 0, 'Sensitivity': 0, 'PPV': 0, 'Dice': 0}
    modals_metrics = dict()
    metrics = ['Miou', 'Sensitivity', 'PPV', 'Dice', 'OR', 'UR']
    for modal in modals:
        modals_metrics[modal] = dict()
        for metric in metrics:
            modals_metrics[modal][metric] = 0


    Dice_Score = 0
    for i in range(val_set_len):
        # 三种模态数据路径
        img_path = dict()
        for modal in modals:
            img_path[modal] = os.path.join(data_dir, modal,'Images', val_img_list[modal][i]+'.npy' )

        Label_path = os.path.join(data_dir, 'T1W\Labels', val_img_list['T1W'][i]+'.npy' )

        # 将图像传入网络，进行预测
        pre_modal = dict()
        for modal in modals:
            pre_modal[modal] = predict_image(img_path[modal],model_dict[modal])

        # 求得分割图
        outputs_modal = dict()
        for modal in modals:
            outputs_modal[modal] = pre_modal[modal].reshape([256, 256])

        Label = Image.fromarray(np.load(Label_path)).convert('L')
        # Dice_Score += Dice_score(Label, outputs, num_classes)

        (Miou_one_img, Sensitivity_one_img, PPV_one_img, Dice_one_img, OR_one_img, UR_one_img), integrate_output = Integrate_predict(pre_modal, Label, num_classes)

        inputs = dict()
        for modal in modals:
            inputs[modal] = np.load(img_path[modal])

        # print("inte:\n",integrate_output)
        label = np.array(Label.convert('RGB'))*255
        r_lab, g_lab, b_lab = label[:,:,0].copy(), label[:,:,1].copy(), label[:,:,2].copy()
        b_lab = b_lab*0
        r_lab = r_lab*0
        # g_lab = g_lab*255
        label = np.dstack([r_lab, g_lab, b_lab])


        for modal in modals:
            outputs_modal[modal] = outputs_modal[modal].astype(np.uint8)
            outputs_modal[modal] = Image.fromarray(outputs_modal[modal])
            outputs_modal[modal] = np.array(outputs_modal[modal].convert('RGB'))*255
            r_modal, g_modal, b_modal = outputs_modal[modal][:, :, 0].copy(), outputs_modal[modal][:, :, 1].copy(), outputs_modal[modal][:, :, 2].copy()
            # b_lab = b_lab * 0
            r_modal = r_modal * 0
            g_modal = g_modal*255
            outputs_modal[modal] = np.dstack([r_modal, g_modal, b_modal])

        integrate_output = integrate_output.astype(np.uint8)
        integrate_output = Image.fromarray(integrate_output)
        integrate_output = np.array(integrate_output.convert('RGB'))
        r_igt, g_igt, b_igt = integrate_output[:,:,0].copy(), integrate_output[:,:,1].copy(), integrate_output[:,:,2].copy()
        r_igt = r_igt*255
        integrate_output = np.dstack([r_igt, g_igt, b_igt])

        # plt.subplots_adjust(wspace=0.01)
        # for col in range(len(modals)):
        #     plt.subplot(4, modal_len, 0 * modal_len + (col+1)) #1 4 7 10
        #     plt.title(modals[col])
        #     plt.imshow(inputs[modals[col]], cmap='gray')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.subplot(4, modal_len, 1 * modal_len + (col+1))
        #     plt.imshow(inputs[modals[col]], cmap='gray', alpha=0.7)
        #     plt.imshow(label, alpha=0.3)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.subplot(4, modal_len, 2 * modal_len + (col+1))
        #     plt.imshow(inputs[modals[col]], cmap='gray', alpha=0.7)
        #     plt.imshow(outputs_modal[modals[col]], alpha=0.3)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.subplot(4, modal_len, 3 * modal_len + (col+1))
        #     plt.imshow(inputs[modals[col]], cmap='gray', alpha=0.7)
        #     plt.imshow(integrate_output,alpha=0.3)
        #     plt.xticks([])
        #     plt.yticks([])
        #     # plt.tight_layout()
        # plt.show()


        Miou += Miou_one_img
        Sensitivity += Sensitivity_one_img
        PPV += PPV_one_img
        Dice += Dice_one_img
        OR += OR_one_img
        UR += UR_one_img

        for modal in modals:
            Miou_value, Sensitivity_value, PPV_value, Dice_value, OR_value, UR_value = hist_operation(num_classes, Label, pre_modal[modal])
            modals_metrics[modal]['Miou'] += Miou_value
            modals_metrics[modal]['Sensitivity'] += Sensitivity_value
            modals_metrics[modal]['PPV'] += PPV_value
            modals_metrics[modal]['Dice'] += Dice_value
            modals_metrics[modal]['OR'] += OR_value
            modals_metrics[modal]['UR'] += UR_value


    Miou, Sensitivity, PPV ,Dice, OR, UR = Miou/(i+1), Sensitivity/(i+1), PPV/(i+1), Dice/(i+1), OR/(i+1), UR/(i+1)
    for modal in modals:
        modals_metrics[modal]['Miou'] = modals_metrics[modal]['Miou'] / (i+1)
        modals_metrics[modal]['Sensitivity'] = modals_metrics[modal]['Sensitivity'] / (i+1)
        modals_metrics[modal]['PPV'] = modals_metrics[modal]['PPV'] / (i+1)
        modals_metrics[modal]['Dice'] = modals_metrics[modal]['Dice'] / (i+1)
        modals_metrics[modal]['OR'] = modals_metrics[modal]['OR'] / (i + 1)
        modals_metrics[modal]['UR'] = modals_metrics[modal]['UR'] / (i + 1)

    Dice_Score = Dice_Score/(i+1)
    print('integrate_Miou:{:.2f}%,integrate_Sensitivity:{:.2f}%,integrate_PPV:{:.2f}%,integrate_Dice:{:.2f}%,OR:{:.2f}%,UR:{:.2f}%'.format(Miou*100, Sensitivity*100, PPV*100,Dice*100,OR*100, UR*100))
    for modal in modals:
        print(modal+" ----- Miou:{:.2f}% , Sensitivity:{:.2f}% , PPV:{:.2f}% , Dice:{:.2f}% , UR:{:.2f}% , OR:{:.2f}% ".\
              format(modals_metrics[modal]['Miou']*100,modals_metrics[modal]['Sensitivity']*100,modals_metrics[modal]['PPV']*100,modals_metrics[modal]['Dice']*100,
                     modals_metrics[modal]['UR']*100,modals_metrics[modal]['OR']*100))







