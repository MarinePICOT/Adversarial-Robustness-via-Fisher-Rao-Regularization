
from __future__ import print_function

import os
from os.path import join
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import load_cfg_from_cfg_file, merge_cfg_from_list, find_free_port, setup, cleanup, get_model

from .losses import trades_loss, noise_loss
from .attacks.attack_pgd import pgd

from .attacks.autoattack import AutoAttack
from torch.utils.tensorboard import SummaryWriter

from .datasets.dataset import get_dataset, SemiSupervisedDataset, SemiSupervisedSampler



import logging


# ----------------------------- CONFIGURATION ----------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg




def eval(args, model, device, eval_set, loader):
    loss = 0
    total = 0
    correct = 0
    adv_correct = 0
    adv_correct_clean = 0
    adv_total = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            data, target = data[target != -1], target[target != -1]
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.distance == 'Linf':
                    # run medium-strength gradient attack
                    is_correct_clean, is_correct_rob = pgd(
                        model, data, target,
                        epsilon=args.epsilon,
                        num_steps=2 * args.perturb_steps,
                        step_size=args.step_size,
                        random_start=False)
                    incorrect_clean = (1-is_correct_clean).sum()
                    incorrect_rob = (1-np.prod(is_correct_rob, axis=1)).sum()
            else:
                    raise ValueError('No support for distance %s',
                                     args.distance)
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

    eval_data = dict(loss=loss, accuracy=accuracy,
                     robust_accuracy=robust_accuracy,
                     robust_clean_accuracy=robust_clean_accuracy)
    eval_data = {eval_set + '_' + k: v for k, v in eval_data.items()}
    logging.info(
        '{}: Clean loss: {:.4f}, '
        'Clean accuracy: {}/{} ({:.2f}%), '
        '{} clean accuracy: {}/{} ({:.2f}%), '
        'Robust accuracy {}/{} ({:.2f}%)'.format(
            eval_set.upper(), loss,
            correct, total, 100.0 * accuracy,
            'Smoothing' if args.distance == 'L2' else 'PGD',
            adv_correct_clean, adv_total, 100.0 * robust_clean_accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))

    return loss, accuracy, robust_accuracy, robust_clean_accuracy




# ----------------------------- TRAINING LOOP ----------------------------------
def main():
    
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    os.makedirs(args.model_dir,exist_ok=True)
    args.distributed = False
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
            logging.StreamHandler()
        ])
    logger = logging.getLogger()

    logging.info('Robust self-training')
    logging.info('Args: %s', args)



    if args.seed is not None:
        cudnn.benchmark = True
    # useful setting for debugging
        #cudnn.benchmark = False
        #cudnn.deterministic = True
        torch.cuda.manual_seed(args.seed)

    
    dataset = get_dataset(args)
    
    if args.unlabeled :
        trainset = SemiSupervisedDataset(args,
                                     add_svhn_extra=args.svhn_extra,
                                     root=args.data_dir, train=True,
                                     download=True,
                                     aux_data_filename=args.aux_data_filename,
                                     add_aux_labels=not args.remove_pseudo_labels,
                                     aux_take_amount=args.aux_take_amount)

        train_batch_sampler = SemiSupervisedSampler(
            trainset.sup_indices, trainset.unsup_indices,
            args.batch_size, args.unsup_fraction,
            num_batches=int(np.ceil(50000 / args.batch_size)))
        epoch_size = len(train_batch_sampler) * args.batch_size

        kwargs = {'num_workers': args.workers, 'pin_memory': True} if len(args.gpus)>0 else {}
        train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)

        testset = SemiSupervisedDataset(args,
                                    root=args.data_dir, train=False,
                                    download=True)
                                    

        test_loader = DataLoader(testset, batch_size=args.val_batch_size,
                                  shuffle=False, **kwargs)
    else :
        train_loader, test_loader, train_sampler = dataset.make_loaders(args)
    
    train_df = pd.DataFrame()
    eval_df = pd.DataFrame()

    model = get_model(args.arch, num_classes=dataset.num_classes,
                      normalize_input=False)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    save_dir = join(args.model_dir, args.data_name)
    state_dict = torch.load(join(save_dir, args.test_filename))

    model.load_state_dict(state_dict)
    model.eval()

    loss,accuracy,robust_accuracy,robust_clean_accuracy = eval(args, model, device, 'test', test_loader)
    
    log_path = os.path.join(args.model_dir,'log_test.txt')
    adversary = AutoAttack(args, model, norm='Linf', eps=args.epsilon,log_path=log_path,version='standard',device=args.device)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                        bs=int(args.val_batch_size))

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                        args.model_dir, 'aa', 'standard', adv_complete.shape[0], args.epsilon))

        else:
                    # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                        y_test[:args.n_ex], bs=int(args.val_batch_size))

            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}.pth'.format(
                        args.model_dir, 'aa', 'standard', adv_complete.shape[0], args.epsilon))
    

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
