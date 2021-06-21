
from __future__ import print_function

import os
from os.path import join
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
from tqdm import tqdm

from .utils import load_cfg_from_cfg_file, merge_cfg_from_list, get_model

from .losses import fire_loss, noise_loss
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



# ------------------------------------------------------------------------------

# ----------------- DATASET WITH AUX PSEUDO-LABELED DATA -----------------------

# ------------------------------------------------------------------------------


# ----------------------- TRAIN AND EVAL FUNCTIONS -----------------------------
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_metrics = []
    bar = tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in bar:
        args.step += 1
        step = args.step
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss, natural_loss, robust_loss, entropy_loss_unlabeled = fire_loss(args,
                model=model,
                x_natural=data,
                y=target,
                optimizer=optimizer,
                epoch=epoch,
                step_size=args.step_size,
                epsilon=args.epsilon,
                perturb_steps=args.perturb_steps,
                beta=args.beta,
                distance=args.distance,
                adversarial=args.distance == 'Linf',
                entropy_weight=args.entropy_weight)


        loss.backward()
        optimizer.step()

        train_metrics.append(dict(
            epoch=epoch,
            loss=loss.item(),
            natural_loss=natural_loss.item(),
            robust_loss=robust_loss.item(),
            entropy_loss_unlabeled=entropy_loss_unlabeled.item()))

        # print progress
        if batch_idx % args.log_interval == 0:
            args.writter.add_scalar('Train_Loss', loss.item(), step)
            args.writter.add_scalar('Train_Natural_Loss', natural_loss.item(), step)
            args.writter.add_scalar('Train_Robust_loss', robust_loss.item(), step)
            logging.info(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    return train_metrics


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
    if adv_total > 0:
        robust_clean_accuracy = adv_correct_clean / adv_total
        robust_accuracy = adv_correct / adv_total
    else:
        robust_accuracy = robust_clean_accuracy = 0.

    logging.info(
        '{}: Clean loss: {:.4f}, '
        '{} clean accuracy: {}/{} ({:.2f}%), '
        'Robust accuracy {}/{} ({:.2f}%)'.format(
            eval_set.upper(), loss,
            'Smoothing' if args.distance == 'L2' else 'PGD',
            adv_correct_clean, adv_total, 100.0 * robust_clean_accuracy,
            adv_correct, adv_total, 100.0 * robust_accuracy))
    if eval_set == 'test' :
        args.writter.add_scalar('Eval_Clean_Loss', loss, args.step)
        args.writter.add_scalar('Eval_Clean_Accuracy', 100.0 * robust_clean_accuracy, args.step)
        args.writter.add_scalar('Eval_Robust_Accuracy', 100.0 * robust_accuracy, args.step)
    else :
        args.writter.add_scalar('Train_Clean_Loss', loss, args.step)
        args.writter.add_scalar('Train_Clean_Accuracy', 100.0 * robust_clean_accuracy, args.step)
        args.writter.add_scalar('Train_Robust_Accuracy', 100.0 * robust_accuracy, args.step)

    return loss, robust_accuracy, robust_clean_accuracy


def adjust_learning_rate(args,optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    schedule = args.lr_schedule
    # schedule from TRADES repo (different from paper due to bug there)
    if schedule == 'trades':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
    # schedule as in TRADES paper
    elif schedule == 'trades_fixed':
        if epoch >= 0.75 * args.epochs:
            lr = args.lr * 0.1
        if epoch >= 0.9 * args.epochs:
            lr = args.lr * 0.01
        if epoch >= args.epochs:
            lr = args.lr * 0.001
    # cosine schedule
    elif schedule == 'cosine':
        lr = args.lr * 0.5 * (1 + np.cos((epoch - 1) / args.epochs * np.pi))
    # schedule as in WRN paper
    elif schedule == 'wrn':
        if epoch >= 0.3 * args.epochs:
            lr = args.lr * 0.2
        if epoch >= 0.6 * args.epochs:
            lr = args.lr * 0.2 * 0.2
        if epoch >= 0.8 * args.epochs:
            lr = args.lr * 0.2 * 0.2 * 0.2
    else:
        raise ValueError('Unkown LR schedule %s' % schedule)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# ------------------------------------------------------------------------------

# ----------------------------- TRAINING LOOP ----------------------------------
def main():
    
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    args.device = device
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
        if data_name=='mnist':
            model = model.cuda()
        else :
            model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    if not os.path.exists('runs_no_unlabeled_{}'.format(args.data_name, args.arch)):
      os.makedirs('runs_no_unlabeled_{}'.format(args.data_name,args.arch))
    tb_writer = SummaryWriter('runs_no_unlabeled_{}/{}_{}'.format(args.data_name,args.arch, args.beta))
    args.writter = tb_writer
    
    max_test_acc = 0.
    args.step = 0
    save_dir = join(args.model_dir, args.data_name)
    os.makedirs(save_dir,exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(args,optimizer, epoch)
        logger.info('Setting learning rate to %g' % lr)
        # adversarial training
        train_data = train(args, model, device, train_loader, optimizer, epoch)
        train_df = train_df.append(pd.DataFrame(train_data), ignore_index=True)

        # evaluation on natural examples
        logging.info(120 * '=')
        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            loss,robust_accuracy,robust_clean_accuracy = eval(args, model, device, 'test', test_loader)
            eval_data = dict(epoch=epoch, loss=loss,
                                 robust_accuracy=robust_accuracy,
                                 robust_clean_accuracy=robust_clean_accuracy)
            eval_data = {'test' + '_' + k: v for k, v in eval_data.items()}

            eval_df = eval_df.append(pd.Series(eval_data), ignore_index=True)
            logging.info(120 * '=')

        # save stats
        train_df.to_csv(os.path.join(save_dir, 'stats_train.csv'))
        eval_df.to_csv(os.path.join(save_dir, 'stats_eval.csv'))

        # save checkpoint
        if robust_accuracy >= max_test_acc:
            max_test_acc = robust_accuracy

            torch.save(model.state_dict(),
                           join(save_dir, f'{args.arch}-best.pt'))
            torch.save(optimizer.state_dict(),
                           join(save_dir, 'optimizer-best.pt'))
                           
        if epoch % args.save_freq == 0 :
            torch.save(model.state_dict(),
                           join(save_dir, f'{args.arch}-{epoch}.pt'))
            torch.save(optimizer.state_dict(),
                           join(save_dir, f'optimizer-{epoch}.pt'))
            if epoch > 1 :
                os.remove(join(save_dir, f'{args.arch}-{epoch-args.save_freq}.pt'))
                os.remove(join(save_dir, f'optimizer-{epoch-args.save_freq}.pt'))


    # Test AutoAttack at the end of training
                        
    log_path_last = os.path.join(args.model_dir,'log_last_{}.txt'.format(args.unsup_fraction))
    adversary = AutoAttack(args, model, norm='Linf', eps=args.epsilon, log_path=log_path_last,version='standard',device=args.device)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex], bs=int(args.val_batch_size))

            torch.save({'adv_complete_last': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}_{}.pth'.format(
                        args.model_dir, 'aa', 'standard', adv_complete.shape[0], args.epsilon,args.unsup_fraction))

        else:
                    # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                        y_test[:args.n_ex], bs=int(args.val_batch_size))

            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_last.pth'.format(
                        args.model_dir, 'aa', 'standard', len(test_loader.dataset), args.epsilon))
        
        
    loss,robust_accuracy,robust_clean_accuracy = eval(args, model, device, 'test', test_loader)
                        
                        
    print('Epoch : {}, Clean Accuracy : {}, Adversarial Accuracy : {} '.format(epoch, robust_clean_accuracy, robust_accuracy) )

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
