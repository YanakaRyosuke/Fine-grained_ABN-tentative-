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

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import cv2
import numpy as np

from models.cifar.res_copy import Net

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
parser.add_argument('--train-batch', default=100, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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
"""
#############################################
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
"""

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



    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    """
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
    """
    bird_dataset = datasets.ImageFolder(root = "./",
                                    transform=data_transform)
###################################################
    n_train = int(len(bird_dataset) * 0.6)
    n_val = len(bird_dataset) *2
    n_test = len(bird_dataset) - n_train - n_val
    
    trainset, valset, testset = torch.utils.data.random_split(
        bird_dataset,
        [n_train, n_val, n_test]
    )
###################################################
    #trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    #trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)

    #testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    #testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size)

    # Model
    print("==> creating model '{}'".format(args.arch))

    model = Net() #モデルの呼び出し、今回は簡易版のため畳み込み3層、convlstm cell１層、全結合から構成される


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
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

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

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device=device), targets.to(device=device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        resize_crops = torch.Tensor(np.zeros((4 ,args.train_batch, 3, 32, 32))).to(device=device)
        #focus = random.randint(1, 5)
        focus = 2
        for rate in range(1, 5, 1): #クロップ処理
            crop = inputs[:,:,focus*rate-1:31-focus*rate,focus*rate-1:31-focus*rate]
            resize_crop = nn.Upsample((32,32), mode='bilinear')
            resize_crops[rate-1] = resize_crop(crop)

        # compute output
        per_outputs, _ = model(resize_crops[0], resize_crops[1], resize_crops[2], resize_crops[3])

        loss = criterion(per_outputs, targets)

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
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.to(device=device), targets.to(device=device)
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        resize_crops = torch.Tensor(np.zeros((4 ,args.train_batch, 3, 32, 32))).to(device=device)
        #focus = random.randint(1, 5)
        focus = 2
        for rate in range(1, 5, 1):
            crop = inputs[:,:,focus*rate-1:31-focus*rate,focus*rate-1:31-focus*rate]
            resize_crop = nn.Upsample((32,32), mode='bilinear')
            resize_crops[rate-1] = resize_crop(crop)

        # compute output
        outputs, attention = model(resize_crops[0], resize_crops[1], resize_crops[2], resize_crops[3])

        '''
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
                #import pdb;pdb.set_trace()
                v_img = ((item_img.transpose((1, 2, 0)) * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]) * 255
                v_img = v_img[:, :, ::-1]
                v_img2 = ((vis_inputs2[i].transpose((1, 2, 0)) * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]) * 255
                v_img2 = v_img2[:, :, ::-1]
                v_img3 = ((vis_inputs3[i].transpose((1, 2, 0)) * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]) * 255
                v_img3 = v_img3[:, :, ::-1]
                v_img4 = ((vis_inputs4[i].transpose((1, 2, 0)) * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]) * 255
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
                img_concat = np.concatenate([v_img, v_img2, v_img3, v_img4], axis=0)
                jet_concat = np.concatenate([jet_map1, jet_map2, jet_map3, jet_map4], axis=0)
                img_jet_concat = np.concatenate([img_concat, jet_concat], axis=1)
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
        '''


        loss = criterion(outputs, targets)


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

def min_max(x, axis=None):
    #min = 0 #x.min(axis=axis, keepdims=True)
    #max = 1 #x.max(axis=axis, keepdims=True)
    min = np.min(x)
    max = np.max(x)
    result = (x-min)/(max-min)
    return result

if __name__ == '__main__':
    main()
