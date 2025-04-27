#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/16 14:35
# @Author : lt,fhh
# @FileName: __init__.py.py
# @Software: PyCharm

import torch
import torch.nn as nn
class BCEFocalLoss(nn.Module):

    def __init__(self, gamma=2, reduction='mean', class_weight=None):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weight = class_weight

    def forward(self, data, label):
        sigmoid = nn.Sigmoid()
        pt = sigmoid(data).detach()
        if self.class_weight is not None:
            label_weight = ((1 - pt) ** self.gamma) * self.class_weight
            # label_weight = torch.exp((1 - pt)) * self.class_weight
            # label_weight = torch.exp((2 * (1 - pt)) ** self.gamma) * self.class_weight
        else:
            label_weight = (1 - pt) ** self.gamma
            # label_weight = torch.exp((1 - pt))
            # label_weight = torch.exp((2 * (1 - pt)) ** self.gamma)

        focal_loss = nn.BCEWithLogitsLoss(weight=label_weight, reduction=self.reduction)
        return focal_loss(data, label)
class FocalDiceLoss(nn.Module):
    """multi-label focal-dice loss"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class CombineLoss(nn.Module):
    """multi-label focal-dice loss"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=None, clip_neg=None, alpha=0.3, beta=0.5, reduction='mean'):
        super(CombineLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss_ce = nn.BCEWithLogitsLoss()(predict, target)
        loss = loss_pos * self.alpha + loss_neg * (1-self.alpha-self.beta) + loss_ce * self.beta

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class ComLoss(nn.Module):
    """multi-label focal-dice loss"""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=None, clip_neg=None, pos_weight=0.5, reduction='mean'):
        super(ComLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 按行相加
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss_dec = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)
        loss = torch.log(loss_dec)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
class ConLoss(torch.nn.Module):
    def __init__(self, temperature=1):
        super(ConLoss, self).__init__()
        self.temperature = temperature
#L [21,256]
    def forward(self, V, L, Y):
        sim = torch.matmul(V, L.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1,
                               keepdim=True)
        sim = sim - sim_max.detach()
        positive_samples_mask = Y == 1
        negative_samples_mask = ~positive_samples_mask
        exp_sim = torch.exp(sim)
        exp_sim_neg = exp_sim * negative_samples_mask.float()
        exp_sim_neg_sum = exp_sim_neg.sum(dim=1, keepdim=True)
        loss = -torch.log((exp_sim * positive_samples_mask.float()).sum(dim=1) / (
                    exp_sim_neg_sum + exp_sim * positive_samples_mask.float()).sum(dim=1)).mean()
        return loss

class AsymmetricLoss(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True,
                 reduction='mean'):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        if self.reduction == 'mean':
            loss = -loss.mean()
        else:
            loss = -loss.sum()
        return loss
class MLDCSLoss(nn.Module):
   #区分了正负样本
    def __init__(self, p=2, alpha=0.5, reduction='mean'):
        super(MLDCSLoss, self).__init__()
        self.p = p
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        pre_pos = predict*((1-predict)**self.p)
        num_pos = torch.sum(torch.mul(pre_pos, target), dim=1)
        den_pos = torch.sum(pre_pos.pow(self.p) + target.pow(self.p), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos

        pre_neg = (1-predict) * (predict ** self.p)
        num_neg = torch.sum(torch.mul(pre_neg, target), dim=1)
        den_neg = torch.sum(pre_neg.pow(self.p) + (1-target).pow(self.p), dim=1)
        loss_neg = 1-(2*num_neg)/den_neg

        loss = self.alpha*loss_pos + (1-self.alpha)*loss_neg
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


