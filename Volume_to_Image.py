import os
import SimpleITK as sitk
import nibabel as nib
import nrrd
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import random
from PIL import Image

def VolumeToImage(patient_path, target_path, modal):
    for label_data in os.listdir(patient_path):
        if 'mask.nii' in label_data:
            label_path = os.path.join(patient_path, label_data)
    # 对label进行ROI检测，若在某层切片出存在ROI，将该切片单独取出
    ori_label = nib.load(label_path).get_fdata()
    ori_label = np.array(ori_label)
    ROI_detect = []
    for slice_num in range(ori_label.shape[2]):
        if len(ori_label[:, :, slice_num][ori_label[:, :, slice_num]==1]) != 0:
            ROI_detect.append(slice_num)
    # print(ori_label.dtype)
    # plt.imshow(ori_label[:,:,1],cmap='gray')
    # plt.show()
    # print(ori_label[:,:,1])
    # 分别以PNG格式保存标签图片和原数据图片
    # print("ROI_detect: ",ROI_detect)
    for img_data in os.listdir(patient_path):
        if modal+'.nii' == img_data:
            img_path = os.path.join(patient_path, img_data)
            ori_data = nib.load(img_path).get_fdata()
            ori_data = np.array(ori_data)
            for slice_num in ROI_detect:
                slice_data = ori_data[:, :, slice_num]
                slice_label = ori_label[:, :, slice_num]
                # img_slice_data = Image.fromarray(slice_data.astype('uint8')).convert('RGB')
                # # img_slice_data.show()
                # img_slice_label = Image.fromarray(slice_label*255).convert('L')
                # # img_slice_label.show()
                img_name = patient_path.split("\\")[-1] + "_" + modal + str(slice_num) + ".npy"
                np.save(target_path+"\\Images\\"+img_name, slice_data)
                np.save(target_path+"\\Labels\\"+img_name, slice_label)


if __name__ == "__main__":
    modals = ['T1W', 'T2W', 'T1WFLAIR', 'T2WFLAIR', 'STIR',
              'PDW', 'PDMapping', 'T1Mapping', 'T2Mapping']
    curr_path = os.getcwd()
    target_path = os.path.join(curr_path, "Breast_Datasets")

    root_dir = os.path.join(curr_path, "Dataset_raw")

    for modal in modals:
        print("正在处理{}模态数据".format(modal))
        patient_num = 0
        modal_dataset_path = os.path.join(target_path, modal)
        img_path = os.path.join(modal_dataset_path, "Images")
        label_path = os.path.join(modal_dataset_path, "Labels")
        imagesets_txt_path = modal_dataset_path + "\\ImageSets\\Segmentation"
        if not os.path.exists(img_path) or not os.path.exists(label_path) or not os.path.exists(imagesets_txt_path):
            os.makedirs(img_path)
            os.makedirs(label_path)
            os.makedirs(imagesets_txt_path)
        for DataFile in os.listdir(root_dir):
            DataFile_path = os.path.join(root_dir, DataFile)
            if os.path.isdir(DataFile_path) and "Part" in DataFile:
                for patient_exa in os.listdir(DataFile_path):
                    patient_exa_path = os.path.join(DataFile_path, patient_exa)
                    if os.path.isdir(patient_exa_path):
                        # for patient_idx in os.listdir(patient_exa_path):
                        #     patient_idx_path = os.path.join(patient_exa_path, patient_idx)
                        #     # print(patient_idx_path)
                        print("正在处理第{}个病例...".format(patient_num))
                        VolumeToImage(patient_exa_path, modal_dataset_path, modal)
                        print("处理完毕")
                        patient_num += 1

        img_list = os.listdir(img_path)
        label_list = os.listdir(label_path)
        img_list.sort()
        img_list = [img.split(".")[0] for img in img_list]
        label_list.sort()
        label_list = [label.split(".")[0] for label in label_list]
        # 设置训练集和测试集比例
        split_ratio = 0.3
        val_set_num = int(0.3 * len(img_list))
        dataset_index = list(range(len(img_list)))
        random.seed(1)
        val_set_index = random.sample(dataset_index, val_set_num)
        val_set = [img_list[i] for i in val_set_index]
        train_set = [train_sample for train_sample in img_list if train_sample not in val_set]
        train_txt_path = imagesets_txt_path+"\\train.txt"
        val_txt_path = imagesets_txt_path+"\\trainval.txt"
        train_txt = open(train_txt_path, "w")
        val_txt = open(val_txt_path, "w")
        for train_line in train_set:
            train_txt.write(train_line + '\n')
        for val_line in val_set:
            val_txt.write(val_line + '\n')
        train_txt.close()
        val_txt.close()







