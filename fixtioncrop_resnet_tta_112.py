#2021_1_7作製最
#データはbirds,使用モデル，resnet.py
#LSTMなしでresnetのみで実装，そもそもLSTMを使う意味は?
#attentionを出力する機構なし
#Finetuning
#cifar_copy_bird2.py参照
#log,eps,model_bestなどは変えること

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
#import models.cifar.vgg_1_4 as vgg_11
import models.cifar.resnet as resnet_18

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
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
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
      rootpath = "./dataset3/CUB-200-2011_rename_fixations"
      #rootpath = "./"
      target_path = os.path.join(rootpath, phase,  "**", "*.jpg") # 最初の**はクラスのディレクトリ
      path_list = []
      # globを利用してサブディレクトリまでファイルパスを格納
      for path in glob.glob("./dataset3/CUB-200-2011_rename_fixations/*/*.jpg"): #階層によって変わる
      #for path in glob.glob("./dataset/CUB-200-2011_rename_fixations/*.jpg"):
        path_list.append(path)
      return path_list
    
    #001.Black_footed_Albatross
    class ImageTransform():
     def __init__(self):
          self.data_transform = transforms.Compose([
                                                    transforms.Resize((256,256)),
                                                    #transforms.RandomHorizontalFlip(p=0.5),
                                                    #transforms.RandomVerticalFlip(p=0.5),
                                                    transforms.ToTensor(),
                                                    #transforms.Normalize((0.1948, 0.2155, 0.1589),(0.2333, 0.2278, 0.26106))
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
            img = img.convert("RGB")
            #torchvision.io.read_imageとかもある
            img_transformed = self.file_transform(img)#前処理クラスでtensorに変換
            label = img_path.split("/")[3] # リストの値を階層によって変える 3
            label = label.split(".")[0] #.以下を削除
            label = int(label) #ラベルを数値にした
            #######################################
            csv_path = img_path.split("/")[4]
            csv_path = csv_path.split(".")[0]
            csv_path = inclusive_index(csv_list, csv_path)
            
            return img_transformed, label, csv_path

    #trainval_dataset = Dataset(file_list = make_datapath_list(phase="train"), transform=ImageTransform())
    trainval_dataset = Dataset(file_list = make_datapath_list(phase="train"), transform=ImageTransform())
    #ここでデータセット制作
    #ここ変えたので注意
    
    csv_list = glob.glob("./dataset3/CUB-200-2011_rename_fixations/*/*_fixtaions.csv")
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
    #train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [n_train, n_val])
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [n_train, n_val])
    #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True  )
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True )
    
    # Model
    print("==> creating model '{}'".format(args.arch))

    model = resnet_18.resnet18() #モデルの呼び出し、今回は簡易版のため畳み込み3層、convlstm cell１層、全結合から構成される
    model.load_state_dict(torch.load("./models/cifar/pretrain/resnet18-5c106cde.pth"), strict=False)

    #import pdb;pdb.set_trace()
    model.fc = nn.Linear(in_features=512, out_features=35, bias=True)
    #(fc): Linear(in_features=512, out_features=1000, bias=True)
    #model.load_state_dict(torch.load("./models/cifar/pretrain/vgg19-dcbb9e9d.pth"), strict=False)
    model = torch.nn.DataParallel(model).to(device=device)
    #import pdb;pdb.set_trace()
    #model.classifier[6] = nn.Linear(4096, 13)
    """
    import pdb;pdb.set_trace()
    
    for p in model.parameters():
        p.requires_grad=False
    model.classifier[6] = nn.Linear(4096, 13)
    """
    """
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 13)    
    )
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    """

    #import pdb;pdb.set_trace()

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
        logger = Logger(os.path.join(args.checkpoint, 'log_fix_resnet_2000_112_tta.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_fix_resnet_2000_112_tta.txt'), title=title)
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
    savefig(os.path.join(args.checkpoint, 'log_fix_resnet_2000_112_tta.eps'))

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

    csv_list = glob.glob("./dataset3/CUB-200-2011_rename_fixations/*/*_fixtaions.csv")
    #csv_list = glob.glob("./dataset/CUB-200-2011_rename_fixations/001.Black_footed_Albatross/*_fixtaions.csv")

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets, csv) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets, csv_path = inputs.to(device=device), targets.to(device=device), csv.to(device=device)
        inputs, targets, csv_name = torch.autograd.Variable(inputs), torch.autograd.Variable(targets), torch.autograd.Variable(csv)
        targets = targets - 1 #絶対必要
        
        targets_4 = []
        x_list = []
        x_numpy = targets.to('cpu').detach().numpy().copy()
        for i in range(len(x_numpy)):
            x = x_numpy[i]
            x_list = [x, x, x, x]
            targets_4.extend(x_list)
        arr_targets_4 = np.array(targets_4)
        targets = torch.tensor(arr_targets_4).to(device)
    
        #import pdb;pdb.set_trace()
        #import pdb;pdb.set_trace()

        #resize_crops = torch.Tensor(np.zeros((4 ,args.train_batch, 3, 32, 32))).to(device=device)
        #focus = random.randint(1, 5)

        #resize_crops_n = np.zeros((1,4,3,224,224))
        con = []

        for batch_num,(inputs_batch) in enumerate(inputs):
            x_pil = transforms.functional.to_pil_image(inputs_batch.cpu())
            #inputs_batchはtensor
            #x_pilはpilに変換（）
            transform_inputs_img = transforms.Resize(216) #pilとしてresize
            #transform_inputs_img = transforms.Resize(224)
            transform_randomverticalflip = transforms.RandomVerticalFlip(p=0.5) #この辺はpilにかけられる
            transform_randomhorizontalflip = transforms.RandomHorizontalFlip(p=0.5)
            #transform_normalize = transforms.Normalize((0.1948, 0.2155, 0.1589),(0.2333, 0.2278, 0.26106))
            origin_img = transform_inputs_img(x_pil) #origin_imgはpil
            #origin_array = numpy.asarray(origin_img) #pilをnumpyに変換
            #d = torch.stack((a,a,a,a),0)
            #a = torch.Size([2, 3])のときtorch.Size([4, 2, 3])となる
            origin_img_a = transform_randomverticalflip(x_pil)
            origin_img_a = transform_randomhorizontalflip(origin_img_a)
            origin_img_a = transform_inputs_img(origin_img_a)
            origin_tensor = transforms.functional.to_tensor(origin_img_a) #pilをtensorに変換([3, 216, 216])
            origin_tensor = torch.unsqueeze(origin_tensor,0) #torch.Size([1, 3, 216, 216])
            csv_num = int(csv_path[batch_num]) #ここあってる？
            
            #csv_pathに変えて
            arr_crop = crop_fixions_aug(origin_img, csv_list[csv_num])#ちゃんと出てくる
            #この時点で正規化はなし
            crop_len = (arr_crop.shape[0])  #リストの何番目か
            x_tensor = arr_crop
            #x_tensor = torch.from_numpy(arr_crop) #torch.Size([crop_pieces_num, 216, 216, 3])
            #x_tensor = x_tensor.permute(0,3,1,2) #torch.Size([2, 3, 216, 216])

            #import pdb;pdb.set_trace()
            if x_tensor.shape[0] == 2:
                torch_list4 = [origin_tensor, origin_tensor,x_tensor ]
                output4 = torch.cat(torch_list4, dim=0)
                #print(output4.shape)
                #print("2  ###############################" )
            elif x_tensor.shape[0] == 3:
                torch_list4 = [origin_tensor, x_tensor ]
                output4 = torch.cat(torch_list4, dim=0)
                #print(output4.shape)
                #print("3  ###############################" )
            elif x_tensor.shape[0] >= 4: #ここからを>=3にすればそれぞれの個数に対応できる
                torch_list4 = [origin_tensor, x_tensor ]
                output4 = torch.cat(torch_list4, dim=0)
                output4 = output4[:4]
                #print(output4.shape) #torch.Size([4, 3, 216, 216])
                #print(">=4  ###############################" )
                #/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_mean_shift.py:231: UserWarning: Binning data failed with provided bin_size=13.829603, using data points as seeds." using data points as seeds." % bin_size)
                #このエラー何だろう
                #output4 = output4.unsqueeze(0)
            con.append(output4)
            resize_crops = torch.cat(con, dim = 0)
            #import pdb;pdb.set_trace()
            #print(resize_crops.shape)
        resize_crops = resize_crops.to(device)
        #import pdb;pdb.set_trace()
        #per_outputs, _ = model(resize_crops)
        #import pdb;pdb.set_trace()
        # compute output
        
        #per_outputs, _ = model(resize_crops[0], resize_crops[1], resize_crops[2], resize_crops[3])
        #import pdb;pdb.set_trace()
        per_outputs = model(resize_crops)

        #import pdb;pdb.set_trace()
        
        loss = criterion(per_outputs, targets)
        #import pdb;pdb.set_trace()
        print("outputs = " + str(torch.argmax(per_outputs)))
        print("targets = " + str(torch.argmax(targets)))
        #これは何?
        #print("targets(380) = " + str(targets))
        #print("csv_path(381) = " + str(csv_path))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(per_outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        #import pdb;pdb.set_trace()

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

    #import pdb;pdb.set_trace()
        ##################################################

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
    csv_list = glob.glob("./dataset3/CUB-200-2011_rename_fixations/*/*_fixtaions.csv")
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
        """
        targets_4 = []
        x_list = []
        x_numpy = targets.to('cpu').detach().numpy().copy()
        for i in range(len(x_numpy)):
            x = x_numpy[i]
            x_list = [x, x, x, x]
            targets_4.extend(x_list)
        arr_targets_4 = np.array(targets_4)
        targets = torch.tensor(arr_targets_4).to(device)
        #import pdb;pdb.set_trace()
        """

        #resize_crops = torch.Tensor(np.zeros((4 ,args.train_batch, 3, 32, 32))).to(device=device)
        #focus = random.randint(1, 5)

        #resize_crops_n = np.zeros((1,4,3,224,224))
        con = []

        for batch_num,(inputs_batch) in enumerate(inputs):
            x_pil = transforms.functional.to_pil_image(inputs_batch.cpu())
            transform_inputs_img = transforms.Resize(216) #pilとしてresize
            #transform_inputs_img = transforms.Resize(224)
            origin_img = transform_inputs_img(x_pil) #origin_imgはpil
            #origin_array = numpy.asarray(origin_img) #pilをnumpyに変換
            #d = torch.stack((a,a,a,a),0)
            #a = torch.Size([2, 3])のときtorch.Size([4, 2, 3])となる
            origin_tensor = transforms.functional.to_tensor(origin_img) #pilをtensorに変換([3, 216, 216])
            origin_tensor = torch.unsqueeze(origin_tensor,0) #torch.Size([1, 3, 216, 216])
            csv_num = int(csv_path[batch_num]) #ここあってる？
            #csv_pathに変えて

            arr_crop = crop_fixions(origin_img, csv_list[csv_num])#ちゃんと出てくる
            #この時点で正規化はなし
            crop_len = (arr_crop.shape[0])  #リストの何番目か
            x_tensor = arr_crop
            #x_tensor = torch.from_numpy(arr_crop) #torch.Size([crop_pieces_num, 216, 216, 3])
            #x_tensor = x_tensor.permute(0,3,1,2) #torch.Size([2, 3, 216, 216])

            if x_tensor.shape[0] == 2:
                torch_list4 = [origin_tensor, origin_tensor,x_tensor ]
                output4 = torch.cat(torch_list4, dim=0)
                #print(output4.shape)
                #print("2  ###############################" )
            elif x_tensor.shape[0] == 3:
                torch_list4 = [origin_tensor, x_tensor ]
                output4 = torch.cat(torch_list4, dim=0)
                #print(output4.shape)
                #print("3  ###############################" )
            elif x_tensor.shape[0] >= 4: #ここからを>=3にすればそれぞれの個数に対応できる
                torch_list4 = [origin_tensor, x_tensor ]
                output4 = torch.cat(torch_list4, dim=0)
                output4 = output4[:4]
            con.append(output4)
            resize_crops = torch.cat(con, dim = 0)
        resize_crops = resize_crops.to(device)

        #import pdb;pdb.set_trace()

        outputs = model(resize_crops)

        outputs = outputs.to('cpu')
        batch_size_num = 8
        outputs_s = torch.split(outputs, 4 ,dim = 0)
        #import pdb;pdb.set_trace()
        con_list = []
        for i in range(batch_size_num):
            con_list2 = torch.zeros(1,35)
            
            for je in outputs_s[i]:
                #import pdb;pdb.set_trace()
                con_list2 += je
            
            check_tensor = con_list2
            con_list2 = con_list2 / 4 #(わるサイズはaugmentation数)
            import pdb;pdb.set_trace()
            con_list.append(con_list2)
        outputs = torch.cat(con_list).to(device)

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

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint_fix_resnet_2000_112_tta.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best_fix_resnet_2000_112_tta.pth.tar'))

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
  image_fixations_point = arr_fixations_point * 216
  #ここの数値は画像サイズに応じて変えること
  return image_fixations_point

def Meanshift_coordinate(point_origin_list):
  point = change_coordinate(point_origin_list)
  bandwidth=estimate_bandwidth(point,quantile=0.4)
  ms=MeanShift(bandwidth=bandwidth,bin_seeding=True)
  ms.fit(point)

  labels = ms.labels_
  cluster_centers = ms.cluster_centers_

  return cluster_centers

def crop_fixions_aug(pil_image, point_origin_list):
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
    #import pdb;pdb.set_trace()
    #crop_image = crop_image.resize((224, 224))
    crop_image = crop_image.resize((216, 216))

    transform_randomverticalflip = transforms.RandomVerticalFlip(p=0.5) #この辺はpilにかけられる
    transform_randomhorizontalflip = transforms.RandomHorizontalFlip(p=0.5)
    crop_image = transform_randomverticalflip(crop_image)
    crop_image = transform_randomhorizontalflip(crop_image)
           
    crop_tensor = transforms.functional.to_tensor(crop_image)
    #print(crop_tensor.shape)
    Y.append(crop_tensor)
  resize_crops_tensor = torch.stack(Y, dim = 0)
  #print(resize_crops_tensor.shape)
  
  return resize_crops_tensor
######################################
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
    #crop_image = crop_image.resize((224, 224))
    crop_image = crop_image.resize((216, 216))
    crop_tensor = transforms.functional.to_tensor(crop_image)
    #print(crop_tensor.shape)
    Y.append(crop_tensor)
  resize_crops_tensor = torch.stack(Y, dim = 0)
  #print(resize_crops_tensor.shape)
  
  return resize_crops_tensor
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
