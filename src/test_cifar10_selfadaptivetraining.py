import argparse
import os
import shutil
import time
import copy
import PIL.Image as Image
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CIFAR10
from losses import get_loss
from models import get_model
from utils import get_scheduler, get_optimizer, accuracy, save_checkpoint, AverageMeter

import logging

from attack_pgd import pgd
from attacks.autoattack import AutoAttack


parser = argparse.ArgumentParser(description='Self-Adaptive Trainingn')
# network
parser.add_argument('--arch', '-a', metavar='ARCH', default='wrn34',
                    help='model architecture')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
# training setting
parser.add_argument('--data-root', help='The directory of data',
                    default='../semisup-adv-master-3/data', type=str)
parser.add_argument('--dataset', help='dataset used to training',
                    default='cifar10', type=str)
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='optimizer for training')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-schedule', default='step', type=str,
                    help='LR decay schedule')
parser.add_argument('--lr-milestones', type=int, nargs='+', default=[75, 90, 100],
                    help='LR decay milestones for step schedule.')
parser.add_argument('--lr-gamma', default=0.1, type=float,
                    help='LR decay gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# loss function
parser.add_argument('--loss', default='trades_sat', help='loss function')
parser.add_argument('--sat-alpha', default=0.9, type=float,
                    help='momentum term of self-adaptive training')
parser.add_argument('--sat-es', default=70, type=int,
                    help='start epoch of self-adaptive training (default 0)')
# adv training
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007, type=float,
                    help='perturb step size')
parser.add_argument('--beta', type=float, default=1.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--eps',type=float,default=1e-12)
parser.add_argument('--test_batch_size',type=int,default=500)
# misc
parser.add_argument('-s', '--seed', default=None, type=int,
                    help='number of data loading workers (default: None)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-freq', default=1, type=int,
                    help='print frequency (default: 1)')
parser.add_argument('--individual', default=False,
                    help='print frequency (default: 1)')
args = parser.parse_args()

args = parser.parse_args()


# settings
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
use_cuda = True
# not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

# setup data loader








def eval(args, model, device, eval_set, loader):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_correct_clean = 0
    adv_total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, X in enumerate(loader):
            data, target = X[0].to(device), X[1].to(device)
            data, target = data[target != -1], target[target != -1]
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
                    # run medium-strength gradient attack
            is_correct_clean, is_correct_rob = pgd(
                        model, data, target,
                        epsilon=args.epsilon,
                        num_steps=2 * args.num_steps,
                        step_size=args.step_size,
                        random_start=False)
            incorrect_clean = (1-is_correct_clean).sum()
            incorrect_rob = (1-np.prod(is_correct_rob, axis=1)).sum()
            adv_correct_clean += (len(data) - int(incorrect_clean))
            adv_correct += (len(data) - int(incorrect_rob))
            adv_total += len(data)
            total += len(data)
    loss /= total
    accuracy = correct / total
    if adv_total > 0:
        robust_clean_accuracy = adv_correct_clean / adv_total
        robust_accuracy = adv_correct / adv_total
    else:
        robust_accuracy = robust_clean_accuracy = 0.



    return loss, accuracy, robust_accuracy, robust_clean_accuracy




def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, 'training.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    logging.info('Robust self-training')
    logging.info('Args: %s', args)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    
    cudnn.benchmark = True
    global best_prec1

    # prepare dataset

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    num_classes = 10
    testset = CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    
    model = get_model(args, num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()
    path = os.path.join(args.save_dir,'cifar10.tar')
    model.load_state_dict(torch.load(path)["state_dict"])
    
    
    loss, accuracy, robust_accuracy, robust_clean_accuracy  = eval(args, model, device, 'test', test_loader)
        
    print('clean accuracy : {}, robust accuracy : {}'.format(robust_clean_accuracy,robust_accuracy))
    
    log_path = os.path.join(args.save_dir,'log_best.txt')

    adversary = AutoAttack(args, model, norm='Linf', eps=args.epsilon,log_path=log_path,version='standard',device=device)
           
    l = [data[0] for data in test_loader]
    x_test = torch.cat(l,0)
    l = [data[1] for data in test_loader]
    y_test = torch.cat(l,0)
    #print(x_test.shape)
           
    with torch.no_grad():
               if not args.individual:
                   adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                               bs=int(args.test_batch_size))

                   torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                               args.save_dir, 'aa', 'standard', adv_complete.shape[0], args.epsilon))

               else:
                           # individual version, each attack is run on all test points
                   adv_complete = adversary.run_standard_evaluation_individual(x_test,
                               y_test, bs=int(args.val_batch_size))

                   torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}_{}.pth'.format(
                               args.save_dir, 'aa', 'standard', len(test_loader.dataset), args.epsilon))
    

    

        
        

        # save checkpoint
        

if __name__ == '__main__':
    main()

