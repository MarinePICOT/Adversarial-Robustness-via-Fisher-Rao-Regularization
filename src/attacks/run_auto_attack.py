import os
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

import sys

sys.path.insert(0, '..')

from models.wideresnet import *
from models.wideresnet_100 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8. / 255.)
    parser.add_argument('--model_path', type=str, default='./model_test.pt')
    parser.add_argument('--n_ex', type=int, default=1000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--dataset',type=str,default='CIFAR-10')

    args = parser.parse_args()

    # load model
    if args.dataset == 'CIFAR-10' :
        model = WideResNet()
    elif args.dataset == 'CIFAR-100' :
        model=WideResNet100()
    elif args.dataset == 'ImageNet' :
        # TO-DO
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt)
    model.cuda()
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    if args.dataset == 'CIFAR10' :
        item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
        test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == 'CIFAR100' :
        item = datasets.CIFAR100(root=args.data_dir,train=False, transform=transform_chain,download=True)
        test_loader = data.DataLoader(item,batch_size = args.batch_size, shuffle=False)
    elif args.dataset == 'ImageNet' :
        # TO-DO
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load attack
    from autoattack import AutoAttack

    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
                           version=args.version)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2

    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                                                             bs=args.batch_size)
            x_test = x_test.to(device)
            adv_complete = adv_complete.to(device)
            pred_nat = model(x_test)
            pred_adv = model(adv_complete)

            acc_nat = pred_nat.eq(y_test.view_as(pred_nat)).sum().item()
            acc_adv = pred_adv.eq(y_test.view_as(pred_adv)).sum().item()

            print('Natural Accuracy : {} %'.format(acc_nat / args.n_ex))
            print('Adversarial Accuracy : {} %'.format(acc_adv / args.n_ex))




            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                                                                        y_test[:args.n_ex], bs=args.batch_size)

            x_test = x_test.to(device)
            adv_complete = adv_complete.to(device)
            pred_nat = model(x_test)
            pred_adv = model(adv_complete)

            acc_nat = pred_nat.eq(y_test.view_as(pred_nat)).sum().item()
            acc_adv = pred_adv.eq(y_test.view_as(pred_adv)).sum().item()

            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))

            x_test = x_test.to(device)
            adv_complete = adv_complete.to(device)
            pred_nat = model(x_test)
            pred_adv = model(adv_complete)

            acc_nat = pred_nat.eq(y_test.view_as(pred_nat)).sum().item()
            acc_adv = pred_adv.eq(y_test.view_as(pred_adv)).sum().item()

            print('Natural Accuracy : {} %'.format(acc_nat / args.n_ex))
            print('Adversarial Accuracy : {} %'.format(acc_adv / args.n_ex))