import torch
from torch import nn
import torch.nn.functional as F
from anchors import *
import numpy as np
from typing import Dict
from functools import partial
import math, pdb, random

def cfg_get(cfg, key, default):
    try:
        return cfg[key]
    except Exception:
        return default

class ZSGLoss(nn.Module):
    def __init__(self, ratios, scales, cfg):
        super().__init__()
        self.cfg = cfg

        self.ratios = ratios
        self.scales = scales

        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']

        # Which loss fucntion to use
        self.use_focal = cfg['use_focal']
        self.use_softmax = cfg['use_softmax']
        self.use_multi = cfg['use_multi']

        self.lamb_reg = cfg['lamb_reg']
        self.specificity_loss_weight = cfg_get(cfg, 'specificity_loss_weight', 0.1)

        self.loss_keys = ['loss', 'cls_ls', 'box_ls']
        if cfg_get(cfg, 'model_name', '') in ['lang_specificity', 'lang_specificity_reinforce']:
            self.loss_keys.append('spec_ls')
        self.anchs = None
        self.get_anchors = partial(
            create_anchors, ratios=self.ratios,
            scales=self.scales, flatten=True)

        self.box_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, out: Dict[str, torch.tensor],
                inp: Dict[str, torch.tensor], model=None) -> Dict[str, torch.tensor]:
        
        feat_sizes = out['feat_sizes']
        num_f_out = out['num_f_out']

        annot = inp['annot']
        att_box = out['att_out']
        reg_box = out['bbx_out']
        
        device = att_box.device

        if len(num_f_out) > 1:
            num_f_out = int(num_f_out[0].item())
        else:
            num_f_out = int(num_f_out.item())

        if self.anchs is None:
            feat_sizes = feat_sizes[:num_f_out, :]
            anchs = self.get_anchors(feat_sizes)
            anchs = anchs.to(device)
            self.anchs = anchs
        else:
            anchs = self.anchs

        matches = simple_match_anchors(anchs, annot, match_thr=self.cfg['matching_threshold'])
        bbx_mask = (matches >= 0)
        ious1 = IoU_values(annot, anchs)
        _, msk = ious1.max(1)
        
        bbx_mask2 = torch.eye(anchs.size(0)).to(device)[msk]
        bbx_mask2 = bbx_mask2 > 0
        bbx_mask2 = bbx_mask2.to(device)
        top1_mask = bbx_mask2

        if not self.use_multi:
            bbx_mask = bbx_mask2
        else:
            bbx_mask = bbx_mask | bbx_mask2
        
        # all clear
        gt_reg_params = bbox_to_reg_params(anchs, annot)
        box_l = self.box_loss(reg_box, gt_reg_params)
        # box_l_relv = box_l.sum(dim=2)[bbx_mask]
        box_l_relv = box_l.sum(dim=2) * bbx_mask.float()
        box_l_relv = box_l_relv.sum(dim=1) / bbx_mask.sum(dim=-1).float()
        box_loss = box_l_relv.mean()
        if box_loss.cpu() == torch.Tensor([float("Inf")]):
            pdb.set_trace()

        att_box = att_box.squeeze(-1)
        att_box_sigm = torch.sigmoid(att_box)

        alpha = self.alpha
        
        clas_loss = self.focal_loss(att_box, bbx_mask, att_box_sigm, alpha)
        
        spec_loss = att_box.new_tensor(0.0)
        if 'specificity_loss' in out:
            spec_loss = out['specificity_loss']
            if spec_loss.numel() > 1:
                spec_loss = spec_loss.mean()
            spec_loss = spec_loss * self.specificity_loss_weight

        out_loss = clas_loss + self.lamb_reg * box_loss + spec_loss
        
        out_dict = {}
        out_dict['loss'] = out_loss
        out_dict['cls_ls'] = clas_loss
        out_dict['box_ls'] = box_loss
        if 'spec_ls' in self.loss_keys:
            out_dict['spec_ls'] = spec_loss

        return out_dict

    def focal_loss(self, att_box, bbx_mask, att_sigm, alpha):
        encoded_tgt = bbx_mask.float()
        ps = att_sigm
        weights = encoded_tgt * (1-ps) + (1-encoded_tgt) * ps
        alphas = ((1-encoded_tgt) * alpha + encoded_tgt * (1 - alpha))

        weights.pow_(self.gamma).mul_(alphas)
        weights = weights.detach()

        clas_loss = F.binary_cross_entropy_with_logits(att_box, bbx_mask.float(), weight=weights, reduction='none')
        clas_loss = clas_loss.sum() / bbx_mask.sum()

        return clas_loss

    def giou_loss(self, pred_boxes, target_boxes, eps=1e-7):
        iou = box_iou(pred_boxes, target_boxes, eps=eps)

        cx1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        cy1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        cx2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        cy2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + eps

        px_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                  (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)

        gt_area = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * \
                  (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

        ix1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        iy1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        ix2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        iy2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        union = px_area + gt_area - inter + eps

        giou = iou - (c_area - union) / c_area

        return (1.0 - giou).mean()

def box_iou(boxes1, boxes2, eps=1e-7):
    """
    boxes1, boxes2: [N, 4], format xyxy
    """

    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)

    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1 + area2 - inter + eps

    return inter / union

def get_default_loss(ratios, scales, cfg):
    return ZSGLoss(ratios, scales, cfg)
