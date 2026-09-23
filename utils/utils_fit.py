import torch
from nets.Unet_Loss import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score, IOUMetric
import cv2
import torch.nn.functional as F

def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, dice_loss, focal_loss, cls_weights, num_classes, modal, Net_used):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0
    total_dice_score = 0
    miou, Sensitivity, PPV = [0] * 3
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            # print("iteration:",iteration,"batch:",batch)
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                # imgs_size: torch.Size([2, 3, 512, 512])
                # pngs_size: torch.Size([2, 512, 512])
                # labels_size: torch.Size([2, 512, 512, 3])每个像素都是独热编码
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model_train(imgs)

            MIOU_Metrics = IOUMetric(num_classes)
            pr = outputs
            pr = F.softmax(pr.permute(0, 2, 3, 1), dim=-1).cpu()
            pr = pr.argmax(axis=-1)
            meitrics = MIOU_Metrics.evaluate(pr, pngs)
            miou += meitrics[0]
            Sensitivity += meitrics[1]
            PPV += meitrics[2]
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice, dice_score = Dice_loss(outputs, labels)
                loss = loss + main_dice
                total_dice_score += dice_score.item()

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'dice_score': total_dice_score / (iteration + 1),
                                'miou': miou / (iteration + 1),
                                'Sensitivity': Sensitivity / (iteration + 1),
                                'PPV': PPV / (iteration + 1)})
            pbar.update(1)

    print('Finish Train')
    train_loss, train_dice, train_miou, train_Sensitivity, train_PPV = \
        total_loss / (iteration + 1), total_dice_score / (iteration + 1), miou / (iteration + 1), Sensitivity / (iteration + 1), PPV / (iteration + 1)
    miou, Sensitivity, PPV = [0] * 3
    total_dice_score = 0
    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

                outputs = model_train(imgs)
                MIOU_Metrics = IOUMetric(num_classes)
                pr = outputs
                pr = F.softmax(pr.permute(0, 2, 3, 1), dim=-1).cpu()
                pr = pr.argmax(axis=-1)
                meitrics = MIOU_Metrics.evaluate(pr, pngs)
                miou += meitrics[0]
                Sensitivity += meitrics[1]
                PPV += meitrics[2]
                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice, dice_score = Dice_loss(outputs, labels)
                    loss = loss + main_dice
                    total_dice_score += dice_score.item()
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

                val_loss += loss.item()
                val_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'f_score': val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'dice_score': total_dice_score / (iteration + 1),
                                'miou': miou / (iteration + 1),
                                'Sensitivity': Sensitivity / (iteration + 1),
                                'PPV': PPV / (iteration + 1)
                                })
            pbar.update(1)

    loss_history.append_loss(total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1))
    print('Finish Validation')
    val_loss, val_dice, val_miou, val_Sensitivity, val_PPV = \
        total_loss / (iteration + 1), total_dice_score / (iteration + 1), miou / (iteration + 1), Sensitivity / (iteration + 1), PPV / (iteration + 1)

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1)))
    torch.save(model.state_dict(), 'logs/'+Net_used+'/'+modal+'/ep%03d-loss%.3f-val_loss%.3f-moiu%.3f-dice%.3f.pth' % (
    (epoch + 1), total_loss / (epoch_step + 1), val_loss / (epoch_step_val + 1), val_miou, val_dice))
    return train_loss, train_dice, train_miou, train_Sensitivity, train_PPV, val_loss, val_dice, val_miou, val_Sensitivity, val_PPV



def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss,
                         focal_loss, cls_weights, num_classes):
    total_loss = 0
    total_f_score = 0

    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                weights = torch.from_numpy(cls_weights)
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                    weights = weights.cuda()

            optimizer.zero_grad()

            outputs = model_train(imgs)
            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # -------------------------------#
                #   计算f_score
                # -------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    loss_history.append_loss(total_loss / (epoch_step + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (total_loss / (epoch_step + 1)))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / (epoch_step + 1)))
