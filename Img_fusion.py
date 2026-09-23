import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.functional as F
import cv2
import os

modals = ['T1W', 'T2W', 'T1W_FLAIR']
dataset_dir_path = os.getcwd()+'\\Breast_Datasets'
fusion_path = os.path.join(dataset_dir_path, '_'.join(modals))
if not os.path.exists(fusion_path):
    # os.makedirs(fusion_path)
    os.makedirs(fusion_path+'\\Images')
    os.makedirs(fusion_path + '\\ImageSets\\Segmentation')
    os.makedirs(fusion_path + '\\Labels')

train_dataset = dict()
val_dataset = dict()
for modal in modals:
    train_path_txt = os.path.join(dataset_dir_path, modal, 'ImageSets\\Segmentation\\train.txt')
    val_path_txt = os.path.join(dataset_dir_path, modal, 'ImageSets\\Segmentation\\trainval.txt')
    with open(train_path_txt) as train_f:
        train_dataset[modal] = [train_sample.split('\n')[0] for train_sample in train_f.readlines()]
    with open(val_path_txt) as val_f:
        val_dataset[modal] = [val_sample.split('\n')[0] for val_sample in val_f.readlines()]

# 获取Label图像路径
# label_path = dataset_dir_path + '\\' + modals[0] + '\\Labels'
len_train_set = len(train_dataset[modals[0]])
len_val_set = len(val_dataset[modals[0]])

train_txt = open(fusion_path + '\\ImageSets\\Segmentation\\train.txt','w')
for i in range(len_train_set):
    temp_imgs_list = []
    for modal in modals:
        temp_img = cv2.imread(os.path.join(dataset_dir_path,modal,'Images',train_dataset[modal][i]+'.png'))
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        temp_imgs_list.append(temp_img)
    label_img = cv2.imread(os.path.join(dataset_dir_path,modals[0],'Labels',train_dataset[modals[0]][i]+'.png'))
    img_merge = cv2.merge(temp_imgs_list)
    cv2.imwrite(fusion_path + '\\Images\\' + str(i) + '.png', img_merge)
    cv2.imwrite(fusion_path + '\\Labels\\' + str(i) + '.png', label_img)
    train_txt.write(str(i)+'\n')
train_txt.close()

val_txt = open(fusion_path + '\\ImageSets\\Segmentation\\trainval.txt','w')
for j in range(len_val_set):
    temp_imgs_list = []
    for modal in modals:
        temp_img = cv2.imread(os.path.join(dataset_dir_path, modal, 'Images', val_dataset[modal][j] + '.png'))
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        temp_imgs_list.append(temp_img)
    label_img = cv2.imread(os.path.join(dataset_dir_path, modals[0], 'Labels', val_dataset[modals[0]][j] + '.png'))
    img_merge = cv2.merge(temp_imgs_list)
    cv2.imwrite(fusion_path + '\\Images\\' + str(i+1+j) + '.png', img_merge)
    cv2.imwrite(fusion_path + '\\Labels\\' + str(i+1+j) + '.png', label_img)
    val_txt.write(str(i+1+j)+'\n')


# img1 = cv2.imread('D:\\Algorithm\\data_process_scripts\\Breast_Datasets\\T1W\Images\\020777_a_T1W1.png')
# img2 = cv2.imread('D:\\Algorithm\data_process_scripts\\Breast_Datasets\\T1W_FLAIR\\Images\\020777_a_T1W_FLAIR1.png')
# img3 = cv2.imread('D:\\Algorithm\\data_process_scripts\\Breast_Datasets\\T2W\\Images\\020777_a_T2W1.png')
# plt.imshow(img1)
# plt.show()
# # 先对图片转灰度图
# img1_L = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2_L = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# img3_L = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
# # plt.imshow(img3_L)
# # plt.show()
# img_merge = cv2.merge([img1_L, img2_L, img3_L])
# fig = plt.figure(figsize=(8,8))
# plt.subplot(221)
# plt.title('T1W(Gray)')
# plt.xticks([])
# plt.yticks([])
# plt.imshow(img1_L)
# plt.subplot(222)
# plt.title('T1W_FLAIR(Gray)')
# plt.xticks([])
# plt.yticks([])
# plt.imshow(img2_L)
# plt.subplot(223)
# plt.title('T2W(Gray)')
# plt.xticks([])
# plt.yticks([])
# plt.imshow(img3_L)
# plt.subplot(224)
# plt.title('Merge(3-channel)')
# plt.xticks([])
# plt.yticks([])
# plt.imshow(img_merge)
# plt.show()
