import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torch.nn.init as init





class ClipAdapter(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        init.xavier_uniform_(self.fc1[0].weight, gain=1.0)
        init.xavier_uniform_(self.fc2[0].weight, gain=1.0)

    def forward(self, x):
        x = self.fc1(x)
        y = self.fc2(x)
        return x, y

class CLIP_Inplanted(nn.Module):
    def __init__(self, clip_model, features):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        for param in self.parameters():
            param.requires_grad = False
        self.features = features
        self.seg_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )
        self.det_adapters = nn.ModuleList( [ClipAdapter(1024, bottleneck=768) for i in range(len(features))] )


    def forward(self, x):
        with torch.no_grad():
            x = self.image_encoder.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            x = torch.cat(
                [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1)


            x = x + self.image_encoder.positional_embedding.to(x.dtype)

            x = self.image_encoder.patch_dropout(x)
            x = self.image_encoder.ln_pre(x)

            x = x.permute(1, 0, 2)

        seg_patch_tokens = []
        det_patch_tokens = []

        for i in range(24):
            if i<self.features[0]:
                with torch.no_grad():
                    x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            else:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:#
                seg_adapt_med, seg_adapt_out = self.seg_adapters[self.features.index(i+1)](x)
                det_adapt_med, det_adapt_out = self.det_adapters[self.features.index(i+1)](x)

                x = 0.8 * x + 0.1 * seg_adapt_out + 0.1 * det_adapt_out

                seg_patch_tokens.append(seg_adapt_med)
                det_patch_tokens.append(det_adapt_med)



        seg_patch_tokens = [seg_patch_tokens[t].permute(1, 0, 2) for t in range(len(seg_patch_tokens))]
        det_patch_tokens = [det_patch_tokens[t].permute(1, 0, 2) for t in range(len(det_patch_tokens))]

        return 0, seg_patch_tokens, det_patch_tokens









