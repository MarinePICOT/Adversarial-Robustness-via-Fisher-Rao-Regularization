"""
Robust training losses. Based on code from
https://github.com/yaodongyu/TRADES
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import pdb


def entropy_loss(unlabeled_logits):
    unlabeled_probs = F.softmax(unlabeled_logits, dim=1)
    return -(unlabeled_probs * F.log_softmax(unlabeled_logits, dim=1)).sum(
        dim=1).mean(dim=0)


def trades_loss(args,model,
                x_natural,
                y,
                optimizer,
                epoch,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                adversarial=True,
                distance='Linf',
                entropy_weight=0):
  """The TRADES KL-robustness regularization term proposed by
       Zhang et al., with added support for stability training and entropy
       regularization"""
  if beta == 0:
    logits = model(x_natural)
    loss = F.cross_entropy(logits, y)
    inf = torch.Tensor([np.inf])
    zero = torch.Tensor([0.])
    return loss, loss, inf, zero
        
  
  else :
    # define KL-loss
#    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()  # moving to eval mode to freeze batchnorm stats
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
    s_nat = model(x_natural).softmax(1).detach()
    if adversarial:
        if distance == 'Linf':
            x_adv += 0.001 * torch.randn(x_natural.shape).to(args.device)

            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                     s_adv = model(x_adv).softmax(1)
                     sqroot_prod = ((s_nat * s_adv) ** 0.5).sum(1)
                     loss_kl = (torch.acos(sqroot_prod - args.eps)**2).mean(0)
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                                  x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError('No support for distance %s in adversarial '
                             'training' % distance)
    else:
        if distance == 'L2':
            x_adv = x_adv + epsilon * torch.randn_like(x_adv)
        else:
            raise ValueError('No support for distance %s in stability '
                             'training' % distance)

    model.train()  # moving to train mode to update batchnorm stats

    # zero gradient
    optimizer.zero_grad()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    logits_adv = F.log_softmax(model(x_adv), dim=1)
    logits_nat = model(x_natural)
    logits_adv = model(x_adv)
    
    s_adv = logits_adv.softmax(1)
    s_nat = logits_nat.softmax(1)

    loss_natural = F.cross_entropy(logits_nat, y, ignore_index=-1)


    sqroot_prod = ((s_nat * s_adv) ** 0.5).sum(1)

    loss_robust = (torch.acos(sqroot_prod - args.eps)**2).mean(0)

    loss = loss_natural + beta * loss_robust

    is_unlabeled = (y == -1)
    if torch.sum(is_unlabeled) > 0:
        logits_unlabeled = logits[is_unlabeled]
        loss_entropy_unlabeled = entropy_loss(logits_unlabeled)
        loss = loss + entropy_weight * loss_entropy_unlabeled
    else:
        loss_entropy_unlabeled = torch.tensor(0)

  return loss, loss_natural, loss_robust, loss_entropy_unlabeled





