#12/26作製最終的に精度は向上しなかった
#bird使用、vgg_12_25.py, VGG最後のFCを変更
#Finetuning
#
'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from PIL import *
import glob
import re
import os
import pdb

from sklearn.cluster import MeanShift,estimate_bandwidth
import csv

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import cv2
import numpy as np

#from models.cifar.vgg_12_25 import Net
import models.cifar.vgg_12_25 as vgg_11

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=50, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()
save_attention_map = False
state = {k: v for k, v in args._get_kwargs()}

if args.evaluate:
    save_attention_map = True
else:
    save_attention_map = False

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#import pdb;pdb.set_trace()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)



        #158 dd
    #My dataset create
    ##############################################
    def make_datapath_list(phase="train"):
      rootpath = "./dataset/CUB-200-2011_rename_fixations"
      #rootpath = "./"
      target_path = os.path.join(rootpath, phase,  "**", "*.jpg") # 最初の**はクラスのディレクトリ
      path_list = []
      # globを利用してサブディレクトリまでファイルパスを格納
      for path in glob.glob("./dataset/CUB-200-2011_rename_fixations/*/*.jpg"): #階層によって変わる
      #for path in glob.glob("./dataset/CUB-200-2011_rename_fixations/*.jpg"):
        path_list.append(path)
      return path_list
    
    #001.Black_footed_Albatross
    class ImageTransform():
     def __init__(self):
          self.data_transform = transforms.Compose([
                                                    transforms.Resize((256,256)),
                                                    transforms.RandomHorizontalFlip(p=0.5),
                                                    transforms.RandomVerticalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1948, 0.2155, 0.1589),(0.2333, 0.2278, 0.26106))
                                                   ])
     def __call__(self, img):
          return self.data_transform(img)

    ############################################
    class Dataset(data.Dataset):
     def __init__(self, file_list, transform=None):
          self.file_list = file_list
          self.file_transform = transform # 前処理クラスのインスタンス
     
     def __len__(self):
          return len(self.file_list)
     
     def __getitem__(self, index):
           img_path = self.file_list[index] # index番目のpath
           img = Image.open(img_path) # index番目の画像ロード
           #import pdb;pdb.set_trace()
           img_transformed = self.file_transform(img) # 前処理クラスでtensorに変換
           #import pdb;pdb.set_trace
           label = img_path.split("/")[3] # リストの値を階層によって変える 3
           label = label.split(".")[0] #.以下を削除
           label = int(label) #ラベルを数値にした
           #import pdb;pdb.set_trace
           #label = convert(label) # クラスを表すディレクトリ文字列から数値に変更
           ################################
           csv_path = img_path.split("/")[4]
           csv_path = csv_path.split(".")[0]
           csv_path = inclusive_index(csv_list, csv_path)

           #return img_transformed, label, img_path, csv_path
           return img_transformed, label, csv_path

    trainval_dataset = Dataset(file_list = make_datapath_list(phase="train"), transform=ImageTransform())
    
    csv_list = glob.glob("./dataset/CUB-200-2011_rename_fixations/*/*_fixtaions.csv")
    #csv_list = glob.glob("./dataset/CUB-200-2011_rename_fixations/001.Black_footed_Albatross/*_fixtaions.csv")
    n_train = int(len(csv_list) * 0.7)
    n_val = len(csv_list) - n_train

    def inclusive_index(lst, purpose):
        for i, e in enumerate(lst):
            if purpose in e: return i

        raise IndexError
    
    batch_size = 8
    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #trainとvalに分ける
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [n_train, n_val])
    #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True  )
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True )
    
    # Model
    print("==> creating model '{}'".format(args.arch))

    model = vgg_11.vgg11() #モデルの呼び出し、今回は簡易版のため畳み込み3層、convlstm cell１層、全結合から構成される
    model.load_state_dict(torch.load("./models/cifar/pretrain/vgg11-bbd30ac9.pth"), strict=False)
    #import pdb;pdb.set_trace()
    #model.load_state_dict(torch.load("./models/cifar/pretrain/vgg19-dcbb9e9d.pth"), strict=False)
    model = torch.nn.DataParallel(model).to(device=device)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        #import pdb;pdb.set_trace()
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        #is_bets = 0
        #bets_acc = 0
        #########################################
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    csv_list = glob.glob("./dataset/CUB-200-2011_rename_fixations/*/*_fixtaions.csv")
    #csv_list = glob.glob("./dataset/CUB-200-2011_rename_fixations/001.Black_footed_Albatross/*_fixtaions.csv")

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets, csv) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets, csv_path = inputs.to(device=device), targets.to(device=device), csv.to(device=device)
        inputs, targets, csv_name = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(csv)
        targets = targets - 1 #絶対必要
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()

        #resize_crops = torch.Tensor(np.zeros((4 ,args.train_batch, 3, 32, 32))).to(device=device)
        #focus = random.randint(1, 5)

        #resize_crops_n = np.zeros((1,4,3,224,224))
        con = []

        for batch_num,(inputs_batch) in enumerate(inputs):
            x_numpy = inputs[batch_num].to('cpu').detach().numpy().copy()
            x_numpy = np.transpose(x_numpy, (1, 2, 0))
            x_numpy = (x_numpy * 255).astype(np.uint8)

            img = Image.fromarray(x_numpy)
            origin_img = img.resize((168, 168))
            origin_img = np.asarray(origin_img)
            origin_img = np.asarray(origin_img, np.float32)
            origin_img = origin_img/np.max(origin_img)

            origin_img = origin_img[np.newaxis,:,:,:]

            csv_num = int(csv_path[batch_num]) #ここあってる？

            arr_crop = crop_fixions(img, csv_list[csv_num])
            crop_len = (arr_crop.shape[0])
            arr_crop = np.asarray(arr_crop, np.float32)

            con_numpy = np.concatenate([origin_img, arr_crop],0)
            if con_numpy.shape[0] == 2:
                con_numpy = np.concatenate([con_numpy, origin_img],0)
                con_numpy = np.concatenate([con_numpy, origin_img],0)
            elif con_numpy.shape[0] == 3:
                con_numpy = np.concatenate([con_numpy, origin_img],0)
            elif con_numpy.shape[0] == 5:
                con_numpy = np.delete(con_numpy, 1, 0) 
            elif con_numpy.shape[0] == 4:
                con_numpy = con_numpy
            elif con_numpy.shape[0] == 6:
                con_numpy = np.delete(con_numpy, 1, 0)
                con_numpy = np.delete(con_numpy, 1, 0)
            elif con_numpy.shape[0] == 7:
                con_numpy = np.delete(con_numpy, 1, 0)
                con_numpy = np.delete(con_numpy, 1, 0)
                con_numpy = np.delete(con_numpy, 1, 0)

            con.append(con_numpy)

        con_n = np.array(con)
        #import pdb;pdb.set_trace()
        con_n = con_n.astype(np.float32)
        #import pdb;pdb.set_trace()
        resize_crops = torch.from_numpy(con_n).clone()
        resize_crops = resize_crops.permute(1,0,4,2,3)
        resize_crops = resize_crops.to(device)

        #import pdb;pdb.set_trace()
        # compute output
        
        per_outputs, _ = model(resize_crops[0], resize_crops[1], resize_crops[2], resize_crops[3])

        loss = criterion(per_outputs, targets)
        #import pdb;pdb.set_trace()
        print("outputs = " + str(torch.argmax(per_outputs)))
        print("targets = " + str(torch.argmax(targets)))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(per_outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    #import pdb;pdb.set_trace()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    count = 0
    model.eval()

    end = time.time()
    csv_list = glob.glob("./dataset/CUB-200-2011_rename_fixations/*/*_fixtaions.csv")
    #csv_list = glob.glob("./dataset/CUB-200-2011_rename_fixations/001.Black_footed_Albatross/*_fixtaions.csv")

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets, csv) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets, csv_path = inputs.to(device=device), targets.to(device=device), csv.to(device=device)
        inputs, targets, csv_name = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(csv)
        targets = targets -1
        #import pdb;pdb.set_trace()

        #resize_crops = torch.Tensor(np.zeros((4 ,args.train_batch, 3, 32, 32))).to(device=device)
        #focus = random.randint(1, 5)

        #resize_crops_n = np.zeros((1,4,3,224,224))
        con = []

        for batch_num,(inputs_batch) in enumerate(inputs):
            x_numpy = inputs[batch_num].to('cpu').detach().numpy().copy()
            x_numpy = np.transpose(x_numpy, (1, 2, 0))
            x_numpy = (x_numpy * 255).astype(np.uint8)

            img = Image.fromarray(x_numpy)
            origin_img = img.resize((168, 168))
            origin_img = np.asarray(origin_img)
            origin_img = np.asarray(origin_img, np.float32)
            origin_img = origin_img/np.max(origin_img)

            origin_img = origin_img[np.newaxis,:,:,:]

            csv_num = int(csv_path[batch_num]) #ここあってる？

            arr_crop = crop_fixions(img, csv_list[csv_num])
            crop_len = (arr_crop.shape[0])
            arr_crop = np.asarray(arr_crop, np.float32)

            con_numpy = np.concatenate([origin_img, arr_crop],0)
            if con_numpy.shape[0] == 2:
                con_numpy = np.concatenate([con_numpy, origin_img],0)
                con_numpy = np.concatenate([con_numpy, origin_img],0)
            elif con_numpy.shape[0] == 3:
                con_numpy = np.concatenate([con_numpy, origin_img],0)
            elif con_numpy.shape[0] == 5:
                con_numpy = np.delete(con_numpy, 1, 0) 
            elif con_numpy.shape[0] == 4:
                con_numpy = con_numpy
            elif con_numpy.shape[0] == 6:
                con_numpy = np.delete(con_numpy, 1, 0)
                con_numpy = np.delete(con_numpy, 1, 0)
            elif con_numpy.shape[0] == 7:
                con_numpy = np.delete(con_numpy, 1, 0)
                con_numpy = np.delete(con_numpy, 1, 0)
                con_numpy = np.delete(con_numpy, 1, 0)

            con.append(con_numpy)

        con_n = np.array(con)
        #import pdb;pdb.set_trace()
        con_n = con_n.astype(np.float32)
        #import pdb;pdb.set_trace()
        resize_crops = torch.from_numpy(con_n).clone()
        resize_crops = resize_crops.permute(1,0,4,2,3)
        resize_crops = resize_crops.to(device)
        # compute output
        # compute output
        outputs, attention = model(resize_crops[0], resize_crops[1], resize_crops[2], resize_crops[3])
        #import pdb;pdb.set_trace()
        #'''
        attention1 = attention[0]
        attention2 = attention[1]
        attention3 = attention[2]
        attention4 = attention[3]

        if save_attention_map == True:
            vis_attention1 = attention1.data.cpu()
            vis_attention1 = vis_attention1.numpy()
            vis_attention2 = attention2.data.cpu()
            vis_attention2 = vis_attention2.numpy()
            vis_attention3 = attention3.data.cpu()
            vis_attention3 = vis_attention3.numpy()
            vis_attention4 = attention4.data.cpu()
            vis_attention4 = vis_attention4.numpy()
            vis_inputs = resize_crops[0].data.cpu()
            vis_inputs = vis_inputs.numpy()
            vis_inputs2 = resize_crops[1].data.cpu()
            vis_inputs2 = vis_inputs2.numpy()
            vis_inputs3 = resize_crops[2].data.cpu()
            vis_inputs3 = vis_inputs3.numpy()
            vis_inputs4 = resize_crops[3].data.cpu()
            vis_inputs4 = vis_inputs4.numpy()

            in_b, in_c, in_y, in_x = vis_inputs.shape
            i = 0
            #import pdb;pdb.set_trace()
            for item_img, item_att in zip(vis_inputs, vis_attention1):
                #import pdb;pdb.set_trace()#(0.1948, 0.2155, 0.1589),(0.2333, 0.2278, 0.26106)) 0.5, 0.5, 0.5
                v_img = ((item_img.transpose((1, 2, 0)) * [0.5, 0.5, 0.5]) + [0.5, 0.5, 0.5]) * 255
                v_img = v_img[:, :, ::-1]
                v_img2 = ((vis_inputs2[i].transpose((1, 2, 0)) * [0.5, 0.5, 0.5]) + [0.5, 0.5, 0.5]) * 255
                v_img2 = v_img2[:, :, ::-1]
                v_img3 = ((vis_inputs3[i].transpose((1, 2, 0)) * [0.5, 0.5, 0.5]) + [0.5, 0.5, 0.5]) * 255
                v_img3 = v_img3[:, :, ::-1]
                v_img4 = ((vis_inputs4[i].transpose((1, 2, 0)) * [0.5, 0.5, 0.5]) + [0.5, 0.5, 0.5]) * 255
                v_img4 = v_img4[:, :, ::-1]

                resize_att1 = cv2.resize(item_att[0], (in_x, in_y))
                resize_att1 = min_max(resize_att1)
                resize_att1 *= 255.
                #import pdb;pdb.set_trace()

                resize_att2 = cv2.resize(vis_attention2[i,0,:], (in_x, in_y))
                resize_att2 = min_max(resize_att2)
                resize_att2 *= 255.

                resize_att3 = cv2.resize(vis_attention3[i,0,:], (in_x, in_y))
                resize_att3 = min_max(resize_att3)
                resize_att3 *= 255.

                resize_att4 = cv2.resize(vis_attention4[i,0,:], (in_x, in_y))
                resize_att4 = min_max(resize_att4)
                resize_att4 *= 255.

                v_img = np.uint8(v_img)
                v_img2 = np.uint8(v_img2)
                v_img3 = np.uint8(v_img3)
                v_img4 = np.uint8(v_img4)
                vis_map1 = np.uint8(resize_att1)
                jet_map1 = cv2.applyColorMap(vis_map1, cv2.COLORMAP_JET)
                jet_map1 = cv2.addWeighted(v_img, 0.5, jet_map1, 0.5, 0)

                vis_map2 = np.uint8(resize_att2)
                jet_map2 = cv2.applyColorMap(vis_map2, cv2.COLORMAP_JET)
                jet_map2 = cv2.addWeighted(v_img2, 0.5, jet_map2, 0.5, 0)

                vis_map3 = np.uint8(resize_att3)
                jet_map3 = cv2.applyColorMap(vis_map3, cv2.COLORMAP_JET)
                jet_map3 = cv2.addWeighted(v_img3, 0.5, jet_map3, 0.5, 0)

                vis_map4 = np.uint8(resize_att4)
                jet_map4 = cv2.applyColorMap(vis_map4, cv2.COLORMAP_JET)
                jet_map4 = cv2.addWeighted(v_img4, 0.5, jet_map4, 0.5, 0)

                img_concat = np.concatenate([v_img, v_img2, v_img3, v_img4], axis=1)
                jet_concat = np.concatenate([jet_map1, jet_map2, jet_map3, jet_map4], axis=1)
                img_jet_concat = np.concatenate([img_concat, jet_concat], axis=0)

                out_dir = os.path.join('outputs_convlstm_cifar10')
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                    #os.makedirs('./output/attention1')
                    #os.makedirs('./output/attention2')
                    #os.makedirs('./output/attention3')
                    #os.makedirs('./output/attention_con')
                    #os.makedirs('./output/raw')
                #out_path = os.path.join(out_dir, 'attention1', '{0:06d}.png'.format(count))
                #cv2.imwrite(out_path, jet_map1)

                #out_path = os.path.join(out_dir, 'attention2', '{0:06d}.png'.format(count))
                #cv2.imwrite(out_path, jet_map2)

                #out_path = os.path.join(out_dir, 'attention3', '{0:06d}.png'.format(count))
                #cv2.imwrite(out_path, jet_map3)

                out_path = os.path.join(out_dir, 'attention_con', '{0:06d}.png'.format(count))
                cv2.imwrite(out_path, img_jet_concat)

                #out_path = os.path.join(out_dir, 'raw', '{0:06d}.png'.format(count))
                #cv2.imwrite(out_path, v_img)
                count += 1
                i += 1
            import pdb;pdb.set_trace()
        #'''


        loss = criterion(outputs, targets)
        #print(torch.max(outputs))
        #print(torch.max(targets))

        #print(attention.min(), attention.max())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def change_coordinate(point_origin_list):
  #入れるのはcsvの値
  with open(point_origin_list) as f:
    reader = csv.reader(f)
    fixations_point = [row for row in reader]
  fixations_point

  #del fixations_point[0]

  fixations_point_np = [[float(item) for item in row] for row in fixations_point] 
  arr_fixations_point = np.array(fixations_point_np)
  image_fixations_point = arr_fixations_point * 256
  return image_fixations_point

def Meanshift_coordinate(point_origin_list):
  point = change_coordinate(point_origin_list)
  bandwidth=estimate_bandwidth(point,quantile=0.4)
  ms=MeanShift(bandwidth=bandwidth,bin_seeding=True)
  ms.fit(point)

  labels = ms.labels_
  cluster_centers = ms.cluster_centers_

  return cluster_centers

def crop_fixions(pil_image, point_origin_list):
  image_crop_size = 112
  cropsize = image_crop_size/2
  Y = []
  for j,i in enumerate(Meanshift_coordinate(point_origin_list)):
    x_point = i[0]
    y_point = i[1]
    left_side = int(x_point - cropsize)
    top = int(y_point - cropsize)
    right_side = int(x_point + cropsize)
    bottom = int(y_point + cropsize)
    crop_image = pil_image.crop((left_side, top, right_side, bottom))
    crop_image = crop_image.resize((168, 168))
    n_data = np.asarray(crop_image)
    #これ必要？ n_dataの最大値で割ることで正規化
    n_data = n_data/np.max(n_data)
    Y.append(n_data)
  arr_meanshift = np.array(Y)
  
  return arr_meanshift
######################################


def min_max(x, axis=None):
    #min = 0 #x.min(axis=axis, keepdims=True)
    #max = 1 #x.max(axis=axis, keepdims=True)
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result

if __name__ == '__main__':
    main()
