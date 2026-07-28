"""
Model file for zsgnet
Author: Arka Sadhu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from typing import Dict, Any
import math, copy, pdb
import numpy as np

from data_loader import get_data
from fpn_resnet import FPN_backbone
from extended_config import cfg as conf
import clip

# from disentangle import DisentangledLearning
from transformer_encoder import *
from multi_fusion import *
from feat_reinforce import *
from lang_specificity import *

def conv2d(ni: int, nf: int, ks: int = 3, stride: int = 1,
           padding: int = None, bias=False) -> nn.Conv2d:
    "Create and initialize `nn.Conv2d` layer. `padding` defaults to `ks//2`."
    if padding is None:
        padding = ks//2
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride,
                     padding=padding, bias=bias)

def conv2d_relu(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None,
                bn: bool = False, bias: bool = False) -> nn.Sequential:

    layers = [conv2d(ni, nf, ks=ks, stride=stride,
                     padding=padding, bias=bias), nn.ReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True))

def cfg_get(cfg, key, default):
    try:
        return cfg[key]
    except Exception:
        return default

class BackBone(nn.Module):
    def __init__(self, encoder: nn.Module, cfg: dict, out_chs=256):
        super().__init__()
        self.device = torch.device(cfg.device)
        self.encoder = encoder
        self.cfg = cfg
        self.out_chs = out_chs
        self.after_init()
        
        self.lang_projection = nn.Linear(512, self.cfg['model_dim']) # 1024 for RN50, 768 for RN50x16, 512 for RN101 and ViT-B/32
        self.np_projection = nn.Linear(512, self.cfg['model_dim'])

        ### Simple fusion
        self.fusion_manner=nn.ModuleList(
            [
                SimpleFusion(v_planes=self.cfg['model_dim']+8, out_planes=self.cfg['model_dim'], q_planes=self.cfg['model_dim']),
                SimpleFusion(v_planes=self.cfg['model_dim']+8, out_planes=self.cfg['model_dim'], q_planes=self.cfg['model_dim']),
                SimpleFusion(v_planes=self.cfg['model_dim']+8, out_planes=self.cfg['model_dim'], q_planes=self.cfg['model_dim']),
                SimpleFusion(v_planes=self.cfg['model_dim']+8, out_planes=self.cfg['model_dim'], q_planes=self.cfg['model_dim']),
                SimpleFusion(v_planes=self.cfg['model_dim']+8, out_planes=self.cfg['model_dim'], q_planes=self.cfg['model_dim']),
                SimpleFusion(v_planes=self.cfg['model_dim']+8, out_planes=self.cfg['model_dim'], q_planes=self.cfg['model_dim'])
            ]
        )

        ## language-guide reinforce
        self.LanguageGuidedReinforce = nn.ModuleList([IntraIntermodalEnhancementModule(self.cfg['model_dim']) for _ in range(6)])

        specificity_learner_cls = AttributeDrivenSpecificityLearner
        specificity_kwargs = dict(
            dim=self.cfg['model_dim'],
            num_object_clusters=cfg_get(self.cfg, 'specificity_num_clusters', 8),
            text_temperature=cfg_get(self.cfg, 'specificity_text_temperature', 0.07),
            cluster_temperature=cfg_get(self.cfg, 'specificity_cluster_temperature', 0.2),
            map_temperature=cfg_get(self.cfg, 'specificity_map_temperature', 0.07)
        )

        self.LanguageSpecificity = nn.ModuleList([specificity_learner_cls(**specificity_kwargs) for _ in range(6)])
        self.last_aux_losses = {}

        self.posterior_fusion = nn.ModuleList([PosteriorStateFeatureFusion(self.cfg['model_dim']) for _ in range(6)])

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=self.cfg['model_dim'] // 2)
        self.pos_proj = nn.ModuleList([nn.Conv2d(self.cfg['model_dim'], self.cfg['model_dim'], 1) for _ in range(6)])
                                                  
    def after_init(self):
        pass

    def num_channels(self):
        raise NotImplementedError

    def concat_we(self, x, we, only_we=False, only_grid=False):
        assert not (only_we and only_grid)
        coord = self._create_coord(we.size(0), x.size(2), x.size(3))

        word_emb_tile = we.view(we.size(0), we.size(1), 1, 1).expand(we.size(0), we.size(1), x.size(2), x.size(3))
        
        return torch.cat((x, word_emb_tile, coord), dim=1)  # grid_tile
    
    def encode_feats(self, inp):
        return self.encoder(inp)

    def encode_lang_seq_feats(self, inp):
        return None, None
    
    def _make_conv(self, input_dim, output_dim, k, stride=1):
        pad = (k - 1) // 2
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, (k, k), padding=(pad, pad), stride=(stride, stride)),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

    def _create_coord(self, batch, height, width):
        # coord = Variable(torch.zeros(batch,8,height,width).cuda())
        xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
        xv_min = (xv.float()*2 - width)/width
        yv_min = (yv.float()*2 - height)/height
        xv_max = ((xv+1).float()*2 - width)/width
        yv_max = ((yv+1).float()*2 - height)/height
        xv_ctr = (xv_min+xv_max)/2
        yv_ctr = (yv_min+yv_max)/2
        hmap = torch.ones(height,width)*(1./height)
        wmap = torch.ones(height,width)*(1./width)
        
        coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
            xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
            xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
            hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
        coord = coord.unsqueeze(0).repeat(batch,1,1,1)
        
        return coord.to(self.device)
    
    def forward(self, inp, we=None, np_token=None, only_we=False, only_grid=False): 
        self.last_aux_losses = {}
        feats = self.encode_feats(inp)
        lang_emb = self.encode_lang_feats(we) 
        np_emb = self.encode_lang_feats(np_token)

        if self.cfg['model_name'] == 'baseline':
            feat_out = [self.concat_we(f, we, only_we=only_we, only_grid=only_grid) for f in feats]

        elif self.cfg['model_name'] == 'simfuse':
            coord_feats = [self._create_coord(we.size(0), x.size(2), x.size(3)) for x in feats]
            lang_feat = self.lang_projection(lang_emb)

            feat_out = []
            for ii, vis_feat in enumerate(feats):
                coord_feat = coord_feats[ii]
                feat_concat = torch.cat([vis_feat, coord_feat], dim=1)
                
                feat_fused = self.fusion_manner[ii](feat_concat, lang_feat)
                
                feat_concated = self.concat_we(feat_fused, lang_feat, only_we=only_we, only_grid=only_grid)
                feat_out.append(feat_concated)

        elif self.cfg['model_name'] == 'lang_specificity':
            coord_feats = [self._create_coord(we.size(0), x.size(2), x.size(3)) for x in feats]
            lang_feat = self.lang_projection(lang_emb)
            np_feat = self.np_projection(np_emb)

            specificity_losses = []
            feat_out = []
            for ii, vis_feat in enumerate(feats):
                specificity_out = self.LanguageSpecificity[ii](vis_feat, lang_feat, np_feat)
                feat_specific = specificity_out['feat']
                specificity_losses.append(specificity_out['specificity_loss'])

                coord_feat = coord_feats[ii]
                feat_concat = torch.cat([feat_specific, coord_feat], dim=1)
                feat_fused = self.fusion_manner[ii](feat_concat, lang_feat)

                feat_concated = self.concat_we(feat_fused, lang_feat, only_we=only_we, only_grid=only_grid)
                feat_out.append(feat_concated)

        elif self.cfg['model_name'] == 'lang_reinforce':
            coord_feats = [self._create_coord(we.size(0), x.size(2), x.size(3)) for x in feats]
            lang_feat = self.lang_projection(lang_emb)
            np_feat = self.np_projection(np_emb)

            feat_out = []
            for ii, vis_feat in enumerate(feats):
                coord_feat = coord_feats[ii]
                enhanced_feat = self.LanguageGuidedReinforce[ii](vis_feat, lang_feat, np_feat)

                feat_concat = torch.cat([enhanced_feat, coord_feat], dim=1)
                multi_feat = self.fusion_manner[ii](feat_concat, lang_feat)

                feat_concated = self.concat_we(multi_feat, lang_feat, only_we=only_we, only_grid=only_grid)
                feat_out.append(feat_concated)

        elif self.cfg['model_name'] == 'lang_specificity_reinforce':
            coord_feats = [self._create_coord(we.size(0), x.size(2), x.size(3)) for x in feats]
            lang_feat = self.lang_projection(lang_emb)
            np_feat = self.np_projection(np_emb)

            specificity_losses = []
            feat_out = []
            for ii, vis_feat in enumerate(feats):
                b, c, h, w = vis_feat.shape
                yy, xx = torch.meshgrid(
                    torch.linspace(0, 1, h, device=vis_feat.device),
                    torch.linspace(0, 1, w, device=vis_feat.device),
                    indexing='ij'
                )
                positions = torch.stack([xx, yy], dim=-1).view(1, h * w, 2).repeat(b, 1, 1)
                pos = self.pos_embed(positions)              # [B, H*W, C]
                pos = pos.transpose(1, 2).view(b, c, h, w)   # [B, C, H, W]
                pos = self.pos_proj[ii](pos)
                vis_feat_pe = vis_feat + pos

                specificity_out = self.LanguageSpecificity[ii](vis_feat_pe, lang_feat, np_feat)
                feat_specific = specificity_out['feat']
                specificity_losses.append(specificity_out['specificity_loss'])
                spec_conf = specificity_out['specificity_score'].view(vis_feat.size(0), 1, 1, 1)
                
                feat_reinforced = self.LanguageGuidedReinforce[ii](vis_feat_pe, lang_feat, np_feat)

                ## posterior fusion
                feat_enhanced = self.posterior_fusion[ii](
                    feat_specific,
                    feat_reinforced,
                    spec_conf=spec_conf,
                    rein_conf=None
                )
                  
                coord_feat = coord_feats[ii]
                feat_concat = torch.cat([feat_enhanced, coord_feat], dim=1)
                feat_fused = self.fusion_manner[ii](feat_concat, lang_feat)

                feat_concated = self.concat_we(feat_fused, lang_feat, only_we=only_we, only_grid=only_grid)
                feat_out.append(feat_concated)
        
        return feat_out
    
class ClipBackbone(BackBone):
    def after_init(self):
        # self.num_chs = self.num_channels()
        self.num_chs = [512, 1024, 2048] # for RN50
        # self.num_chs = [768, 1536, 3072]
        self.fpn = FPN_backbone(self.num_chs, self.cfg, feat_size=self.out_chs)
    
    def encode_feats(self, inp):
        with torch.no_grad():
            feature_map = self.encoder.encode_image(inp)
        
        x2 = feature_map[2]
        x3 = feature_map[3]
        x4 = feature_map[4]

        feats = self.fpn([x2, x3, x4])

        return feats

    def encode_lang_feats(self, inp):
        if inp is None:
            return None
        with torch.no_grad():
            lang_feats = self.encoder.encode_text(inp.squeeze(1).long())
            lang_feats = lang_feats / lang_feats.norm(dim=-1, keepdim=True)

        return lang_feats

    def encode_lang_seq_feats(self, inp):
        if inp is None:
            return None, None
        if not all(hasattr(self.encoder, attr) for attr in ['token_embedding', 'positional_embedding', 'transformer', 'ln_final', 'text_projection']):
            return None, None

        with torch.no_grad():
            text = inp.squeeze(1).long()
            dtype = self.encoder.token_embedding.weight.dtype

            x = self.encoder.token_embedding(text).to(dtype)
            x = x + self.encoder.positional_embedding.to(dtype)
            x = x.permute(1, 0, 2).contiguous()
            x = self.encoder.transformer(x)
            x = x.permute(1, 0, 2).contiguous()
            x = self.encoder.ln_final(x).to(dtype)
            x = x @ self.encoder.text_projection
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)

            lang_mask = text.ne(0)
            if lang_mask.size(1) > 0:
                lang_mask[:, 0] = False
            eot_idx = text.argmax(dim=-1)
            lang_mask.scatter_(1, eot_idx.unsqueeze(1), False)

        return x, lang_mask
        
class ZSGNet(nn.Module):
    def __init__(self, backbone, n_anchors=1, final_bias=0., cfg=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.device = torch.device(cfg.device)
        self.n_anchors = n_anchors

        self.emb_dim = 1024 if cfg['mdl_to_use'] == 'clip' else 300
        self.bid = cfg['use_bidirectional']
        self.lstm_dim = cfg['lstm_dim'] * 4 if cfg['mdl_to_use'] == 'clip' else cfg['lstm_dim']
        # self.use_late_branch_fusion = self.cfg["model_name"] == "lang_disen_reinforce_late"

        if self.cfg["model_name"] == "baseline":
            self.start_dim_head = self.lstm_dim*(self.bid+1)+256+8  # 1288, + 256 for RN50 (1538), RN50x16 (1282), RN101 (1026)
        else:
            self.start_dim_head = self.lstm_dim + 8

        if self.cfg['use_same_atb']:
            bias = torch.zeros(5 * self.n_anchors)
            bias[torch.arange(4, 5 * self.n_anchors, 5)] = -4
            self.att_reg_box = self._head_subnet(
                5, self.n_anchors, final_bias=bias,
                start_dim_head=self.start_dim_head
            )

        else:
            self.att_box = self._head_subnet(1, self.n_anchors, -4., start_dim_head=self.start_dim_head)
            self.reg_box = self._head_subnet(4, self.n_anchors, start_dim_head=self.start_dim_head)
                
        self.after_init()
        
    def after_init(self):
        "Placeholder if any child class needs something more"
        pass

    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256, start_dim_head=256):            
        layers = [conv2d_relu(start_dim_head, chs, bias=True)]
        layers += [conv2d_relu(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        
        return nn.Sequential(*layers)

    def permute_correctly(self, inp, outc):
        out = inp.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, outc)
        
        return out

    def forward(self, inp: Dict[str, Any]):
        inp0 = inp['img']
        inp1 = inp['qvec']
        inp2 = inp['npvec']
    
        req_emb = inp1.squeeze(1)
        np_emb = inp2.squeeze(1)

        feat_fused = self.backbone(inp0, req_emb, np_emb)

        att_bbx_out = torch.cat([self.permute_correctly(self.att_reg_box(feature), 5) for feature in feat_fused], dim=1)
        att_out = att_bbx_out[..., [-1]]
        bbx_out = att_bbx_out[..., :-1]
        
        out_device = feat_fused[0].device
        feat_sizes = torch.tensor([[f.size(2), f.size(3)] for f in feat_fused], device=out_device)
        num_f_out = torch.tensor([len(feat_fused)], device=out_device)
        
        out_dict = {}
        out_dict['att_out'] = att_out
        out_dict['bbx_out'] = bbx_out
        out_dict['feat_sizes'] = feat_sizes
        out_dict['num_f_out'] = num_f_out

        aux_losses = getattr(self.backbone, 'last_aux_losses', {})
        if isinstance(aux_losses, dict):
            for key, value in aux_losses.items():
                out_dict[key] = value
        
        return out_dict

def get_default_net(num_anchors=1, cfg=None):
    clip_model, _ = clip.load('RN101')
    backbone = ClipBackbone(clip_model, cfg)

    zsg_net = ZSGNet(backbone, num_anchors, cfg=cfg)
    return zsg_net

if __name__ == '__main__':
    # torch.manual_seed(0)
    cfg = conf
    cfg.mdl_to_use = 'ssd_vgg'
    cfg.ds_to_use = 'refclef'
    cfg.num_gpus = 1
    # cfg.device = 'cpu'
    device = torch.device(cfg.device)
    data = get_data(cfg)

    zsg_net = get_default_net(num_anchors=9, cfg=cfg)
    zsg_net.to(device)

    batch = next(iter(data.train_dl))
    for k in batch:
        batch[k] = batch[k].to(device)
    out = zsg_net(batch)
