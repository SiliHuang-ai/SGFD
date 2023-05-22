import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from captum.attr import GuidedBackprop


def set_parameters(env_device,init_steps):
    global device,init_step
    device=env_device
    init_step=init_steps

def comput_attr(input,model,step):
    gbp = GuidedBackprop(model)
    attribution = (gbp.attribute(input, target=0) + gbp.attribute(input, target=1) + gbp.attribute(input,target=2) + gbp.attribute(input, target=3)) / 4
    attr_probs= torch.softmax(attribution.abs().mean(dim=0,keepdim=True),dim=-1)*7
    attr_matrix = torch.matmul(attr_probs.t(),attr_probs).detach()
    return attr_matrix

def weight_learner(cfeatures, pre_features, pre_weight1, min_loss, args, global_weight, step, task_dir, model):
    if pre_weight1 is None:
        pre_features=torch.zeros(cfeatures.size()).to(device)
        pre_weight1=torch.ones(cfeatures.size()[0], 1).to(device)
    adjusted_ratio=0.1 if global_weight else 1.0
    softmax = nn.Softmax(0)
    weight = Variable(torch.ones(cfeatures.size()[0], 1).to(device))#128*1
    weight.requires_grad = True
    cfeaturec = Variable(torch.FloatTensor(cfeatures.size()).to(device))
    cfeaturec.data.copy_(cfeatures.data)
    attr_matrix = comput_attr(cfeaturec,model,step)
    all_feature = torch.cat([cfeaturec, pre_features.detach()], dim=0)
    optimizerbl = torch.optim.SGD([weight], lr=args.lrbl, momentum=0.9)
    for epoch in range(args.epochb):
        lr_setter(optimizerbl, epoch, args, bl=True)
        all_weight = torch.cat((weight, pre_weight1.detach()), dim=0)
        optimizerbl.zero_grad()
        lossb,f_featrues = lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum, attr_matrix)
        lossp = softmax(weight).pow(args.decay_pow).sum()
        #TODO 注意在RL中的衰减
        lambdap = args.lambdap * max((args.lambda_decay_rate ** ((step//init_step) // args.lambda_decay_epoch)),
                                     args.min_lambda_times)
        lossg = lossb/lossb.detach() + 5*lossp/lossp.detach()
        if step == init_step:
            lossg = lossg * args.first_step_cons

        lossg.backward(retain_graph=True)
        optimizerbl.step()

    if step%1000==0:
        all_weight = torch.cat((weight.detach(), pre_weight1.detach()), dim=0)
        lossb, f_featrues = lossb_expect(all_feature, softmax(all_weight), args.num_f, args.sum, attr_matrix)
        if lossb.data<min_loss.data:
            min_loss = lossb.data.detach()
            _plt_cov(f_featrues, all_feature, softmax(all_weight), args.num_f, args.sum, task_dir)
            softmax_weights = list(np.reshape(softmax(all_weight).detach().to('cpu'), -1))
            _plt_weight(softmax_weights, task_dir)


    if step < init_step+10:
        pre_features = (pre_features * (step-init_step) + cfeatures) / (step-init_step + 1)
        pre_weight1 = (pre_weight1 * (step-init_step) + weight) / (step-init_step + 1)

    elif cfeatures.size()[0] < pre_features.size()[0]:
        pre_features[:cfeatures.size()[0]] = pre_features[:cfeatures.size()[0]] * args.presave_ratio + cfeatures * (
                    1 - args.presave_ratio)
        pre_weight1[:cfeatures.size()[0]] = pre_weight1[:cfeatures.size()[0]] * args.presave_ratio + weight * (
                    1 - args.presave_ratio)
    else:
        pre_features = pre_features * args.presave_ratio + cfeatures * (1 - args.presave_ratio)
        pre_weight1 = pre_weight1 * args.presave_ratio + weight * (1 - args.presave_ratio)

    softmax_weight = softmax(weight)

    return softmax_weight, weight ,pre_features, pre_weight1, min_loss

def _plt_cov(f_featrues, all_features, weight, num_f, sum, task_dir):
    cfeaturecs = f_featrues
    np.save(task_dir + '/f_features', cfeaturecs.to('cpu'))
    np.save(task_dir + '/all_features', all_features.to('cpu'))


def _plt_weight(weights, task_dir, num=0):
    plt.figure()
    plt.cla()
    plt.plot([idx for idx in range(1, len(weights) + 1)], weights)
    plt.xlabel('sample_idx')
    plt.ylabel('weights')
    plt.savefig(task_dir + '/weights_plt_{}.png'.format(num), format='png')
    np.save(task_dir + '/softmax_weights', weights)


def lr_setter(optimizer, epoch, args, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = args.lr
    if bl:
        lr = args.lrbl * (0.1 ** (epoch // (args.epochb * 0.5)))
    else:
        if args.cos:
            lr *= ((0.01 + math.cos(0.5 * (math.pi * epoch / args.epochs))) / 1.01)
        else:
            if epoch >= args.epochs_decay[0]:
                lr *= 0.1
            if epoch >= args.epochs_decay[1]:
                lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def lossb_expect(cfeaturec, weight, num_f, sum=True, attr_matrix=None):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).to(device)
    loss = Variable(torch.FloatTensor([0]).to(device))
    weight = weight.to(device)
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]

        cov1 = cov(cfeaturec, weight)
        cov_matrix = cov1 * cov1 * attr_matrix
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss,cfeaturecs

def cov(x, w=None):
    if w is None:
        n = x.shape[0]
        cov = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())
    else:
        w = w.view(-1, 1)
        cov = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        res = cov - torch.matmul(e, e.t())

    return res

def lossb_expect2(cfeaturec, weight, num_f, sum=True):
    cfeaturecs = random_fourier_features_gpu(cfeaturec, num_f=num_f, sum=sum).to(device)
    loss = Variable(torch.FloatTensor([0]).to(device))
    weight = weight.to(device)
    for i in range(cfeaturecs.size()[-1]):
        cfeaturec = cfeaturecs[:, :, i]

        cov1 = cov2(cfeaturec, weight)
        cov_matrix = cov1 * cov1
        loss += torch.sum(cov_matrix) - torch.trace(cov_matrix)

    return loss,cfeaturecs

def cov2(x, w=None):
    if w is None:
        n = x.shape[0]
        xy = torch.matmul(x.t(), x) / n
        e = torch.mean(x, dim=0).view(-1, 1)
        x_y_= torch.matmul(e, e.t())
        e_tile=torch.tile(e,dims=(e.shape[0],n))
        xy_and_x_y= torch.matmul(e_tile, x)
        cov = xy - 2*xy_and_x_y + x_y_
    else:
        n = x.shape[0]
        w = w.view(-1, 1)
        xy = torch.matmul((w * x).t(), x)
        e = torch.sum(w * x, dim=0).view(-1, 1)
        x_y_ = torch.matmul(e, e.t())
        e_tile = torch.tile(e, dims=(1, n))
        xy_and_x_y = torch.matmul(e_tile, w * x)
        cov = xy/2 - xy_and_x_y + x_y_/2

    return cov

def random_fourier_features_gpu(x, w=None, b=None, num_f=None, sum=True, sigma=None, seed=None):
    if num_f is None:
        num_f = 1
    n = x.size(0)
    r = x.size(1)
    x = x.view(n, r, 1)
    c = x.size(2)
    if sigma is None or sigma == 0:
        sigma = 1
    if w is None:
        w = 1 / sigma * (torch.randn(size=(num_f, c)))
        b = 2 * np.pi * torch.rand(size=(r, num_f))
        b = b.repeat((n, 1, 1))

    Z = torch.sqrt(torch.tensor(2.0 / num_f).to(device))

    mid = torch.matmul(x.to(device), w.t().to(device))

    mid = mid + b.to(device)
    mid -= mid.min(dim=1, keepdim=True)[0]
    mid /= mid.max(dim=1, keepdim=True)[0].to(device)
    mid *= np.pi / 2.0

    if sum:
        Z = Z * (torch.cos(mid).to(device) + torch.sin(mid).to(device))
    else:
        Z = Z * torch.cat((torch.cos(mid).to(device), torch.sin(mid).to(device)), dim=-1)

    return Z