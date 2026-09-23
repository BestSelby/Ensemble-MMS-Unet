import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader


from nets.unet import Unet
from nets.SegNet_VGG16 import SegNet
from nets.VGG16_Deeplab_v3plus import VGG16_DeepLabv3_plus
from nets.Unet_Loss import weights_init
from utils.callbacks import LossHistory
from utils.dataloader import UnetDataset,unet_dataset_collate
from utils.utils_fit import fit_one_epoch_no_val,fit_one_epoch
from nets.vgg import VGG16
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary

if __name__ == "__main__":
    Cuda = True
    # 前景+背景
    num_classes = 2

    backbone = "vgg"
    # 是否启用预训练权重
    pretrained = True

    model_path = ""

    # 输入图片大小
    input_shape = [256,256]

    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    Init_Epoch = 0
    Freeze_Epoch = 200
    Freeze_batch_size = 2
    Freeze_lr = 1e-4

    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    UnFreeze_Epoch = 200
    UnFreeze_Batch_size = 2
    UnFreeze_lr = 1e-5

    DataSet_Path = "Breast_Datasets"
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    dice_loss = True
    #   是否使用focal loss来防止正负样本不平衡
    focal_loss = False
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    cls_weights = np.ones([num_classes],np.float32)
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    Freeze_Train = True
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    num_workers = 0

    Net_used = "DeepLabV3+"
    # modals = ['T1W', 'T2W', 'T1WFLAIR', 'T2WFLAIR', 'STIR',
    #           'PDW', 'PDMapping', 'T1Mapping', 'T2Mapping']
    modals = ['T1W', 'T2W', 'T1WFLAIR','T2WFLAIR']

    # modals.sort()

    for modal in modals:
        # model = Unet(num_classes=2, pretrained=pretrained, backbone=backbone).train()
        model = VGG16_DeepLabv3_plus(nInputChannels=3,n_classes=2).train()
        # model = SegNet(class_num=2)
        if not pretrained:
            weights_init(model)
        if model_path != '':
            print('Load weights {}.'.format(model_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k] == np.shape(v))}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        model_train = model.train()
        if Cuda:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
        loss_history = LossHistory("logs/"+Net_used+'/'+modal+'/', val_loss_flag=True)
        print('正在对{}模态数据进行训练'.format(modal))
        # 读取数据集对应的txt
        with open(os.path.join(DataSet_Path, modal, "ImageSets/Segmentation/train.txt"),"r") as f:
            train_lines = f.readlines()

        with open(os.path.join(DataSet_Path, modal, "ImageSets/Segmentation/trainval.txt"),"r") as f:
            val_lines = f.readlines()

        # ------------------------------------------------------#
        #   主干特征提取网络特征通用，冻结训练可以加快训练速度
        #   也可以在训练初期防止权值被破坏。
        #   Init_Epoch为起始世代
        #   Interval_Epoch为冻结训练的世代
        #   Epoch总训练世代
        #   提示OOM或者显存不足请调小Batch_size
        # ------------------------------------------------------#
        if True:
            batch_size = Freeze_batch_size
            lr = Freeze_lr
            start_epoch = Init_Epoch
            end_epoch = Freeze_Epoch

            epoch_step = len(train_lines) // batch_size
            epoch_step_val = len(val_lines) // batch_size
            if epoch_step == 0:
                raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

            optimizer = optim.Adam(model_train.parameters(), lr)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

            train_dataset = UnetDataset(train_lines, modal, input_shape, num_classes, True, DataSet_Path)
            # print("train_dataset:",train_dataset)
            val_dataset = UnetDataset(val_lines, modal, input_shape, num_classes, False, DataSet_Path)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                 drop_last=True, collate_fn=unet_dataset_collate)
            # ------------------------------------#
            #   冻结一定部分训练
            # ------------------------------------#
            if Freeze_Train:
                model.freeze_backbone()
            one_metrics_logs = os.path.join('./metrics_logs', Net_used, modal)
            if not os.path.exists(one_metrics_logs):
                os.makedirs(one_metrics_logs)
            para_log_path = os.path.join(os.getcwd(),"logs",Net_used, modal)
            if not os.path.exists(para_log_path):
                os.makedirs(para_log_path)
            writer = SummaryWriter(log_dir=one_metrics_logs)
            for epoch in range(start_epoch, end_epoch):
                train_loss, train_dice, train_miou, train_Sensitivity, train_PPV, val_loss, val_dice, val_miou, val_Sensitivity, val_PPV = \
                fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                              epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, modal, Net_used)

                writer.add_scalars(main_tag='Loss', tag_scalar_dict={'train_loss': train_loss,
                                                                    'val_loss':val_loss}, global_step=epoch)
                writer.add_scalars(main_tag='Metrics/Dice', tag_scalar_dict={'train_dice':train_dice,
                                                                             'val_dice':val_dice}, global_step=epoch)
                writer.add_scalars(main_tag='Metrics/Miou', tag_scalar_dict={'train_miou': train_miou,
                                                                             'val_miou': val_miou}, global_step=epoch)
                writer.add_scalars(main_tag='Metrics/Sensitivity', tag_scalar_dict={'train_Sensitivity': train_Sensitivity,
                                                                                    'val_Sensitivity': val_Sensitivity}, global_step=epoch)
                writer.add_scalars(main_tag='Metrics/PPV', tag_scalar_dict={'train_PPV': train_PPV,
                                                                            'val_PPV': val_PPV}, global_step=epoch)

                lr_scheduler.step()

            writer.close()



