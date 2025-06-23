#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.frcnn_v2 import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import FRCNNDataset, frcnn_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda            = True
    train_gpu       = [0, 1, 4]
    fp16            = True
    classes_path    = './faster-rcnn-pytorch-master/model_data/voc_classes.txt'
    model_path      = '/data_lg/keru/project/Faster-RCNN-Pytorch/faster-rcnn-pytorch-master/model_data/voc_weights_resnet.pth'
    # model_path      = '/data_lg/keru/project/Faster-RCNN-Pytorch/logs_v2/last_epoch_weights.pth'
    input_shape     = [640, 640]
    backbone        = "resnet50"
    pretrained      = True
    anchors_size    = [4, 16, 32]
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    UnFreeze_Epoch      = 170
    Unfreeze_batch_size = 32
    Freeze_Train        = True
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    save_period         = 1000
    save_dir            = 'logs_v2'
    eval_flag           = True
    eval_period         = 5
    num_workers         = 8

    train_annotation_path   = './faster-rcnn-pytorch-master/map_out/2007_train.txt'
    val_annotation_path     = './faster-rcnn-pytorch-master/map_out/2007_val.txt'

    class_names, num_classes = get_classes(classes_path)
    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)

    model = FasterRCNN(num_classes, anchor_scales = anchors_size, backbone = backbone, pretrained = pretrained)
    if not pretrained:
        weights_init(model)

    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示,head部分没有载入是正常现象,Backbone部分没有载入是错误的。\033[0m")

    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, 
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, 
        Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, 
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, 
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            # for param in model.extractor.parameters():
            for param in model_train.module.extractor.parameters():
                param.requires_grad = False
        # model.freeze_bn()
        model_train.module.freeze_bn()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 16
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小,无法继续进行训练,请扩充数据集。")

        train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
        val_dataset   = FRCNNDataset(val_lines, input_shape, train=False)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=frcnn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=frcnn_dataset_collate)

        train_util = FasterRCNNTrainer(model_train, optimizer)
        eval_callback = EvalCallback(model_train, input_shape, class_names, num_classes, val_lines, log_dir, Cuda,
                                     eval_flag=eval_flag, period=eval_period)

        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                # for param in model.extractor.parameters():
                for param in model_train.module.extractor.parameters():
                    param.requires_grad = True
                # model.freeze_bn()
                model_train.module.freeze_bn()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小,无法继续进行训练,请扩充数据集。")
                gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                 drop_last=True, collate_fn=frcnn_dataset_collate)
                gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                     drop_last=True, collate_fn=frcnn_dataset_collate)
                UnFreeze_flag = True

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model, train_util, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, 
                          gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir)
        loss_history.writer.close()