import torch
import torch.nn as nn
import numpy as np


class BaseNetWithFC(nn.Module):
    def __init__(self, backbone, embedding_dim, n_class, backbone_name=None):
        super(BaseNetWithFC, self).__init__()

        self.backbone = backbone

        self.embedding_dim = embedding_dim
        self.n_class = n_class

        self.fc = nn.Linear(embedding_dim, n_class)

        self.backbone_name = backbone_name



    def forward(self, xs, return_features=False):
        scores = []
        if return_features:
            features = []

        for modality in xs:
            x = xs[modality]
            B, I, C, H, W = x.size()

            x = x.view(B * I, C, H, W)
            x = self.forward_features(x)
            if return_features:
                feature = x.view(B, I, -1, self.embedding_dim)

            x = x[:, 0]  # cls token
            score = self.fc(x)
            score = score.view(B, I, self.n_class)

            score = torch.max(score, dim=1)[0]
            if return_features:
                feature = torch.max(feature, dim=1)[0]

            scores.append(score)
            if return_features:
                features.append(feature)

        scores = torch.mean(torch.stack(scores, dim=0), dim=0)
        if return_features:
            features = torch.mean(torch.stack(features, dim=0), dim=0)
            return scores, features
        return scores

    def forward_features(self, x):
        if self.backbone_name in ['dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g']:

            x = self.backbone.forward_features(x)
            x_norm_clstoken = x['x_norm_clstoken']
            x_norm_regtokens = x['x_norm_regtokens']
            x_norm_patchtokens = x['x_norm_patchtokens']

            x_norm_clstoken = torch.unsqueeze(x_norm_clstoken, 1)
            x = torch.concatenate([x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens], dim=1)

        elif self.backbone_name in ['openclip_vit_b', 'openclip_vit_l']:
            x = self.backbone.forward_features(x)

        else:
            raise Exception(self.backbone_name)

        return x

    def set_grad(self, transfer_type):
        print('transfer', transfer_type)
        self.transfer_type = transfer_type
        if transfer_type == 'finetune':
            pass

        elif transfer_type == 'linear':
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif transfer_type == 'mor':
            # MoR: train the LoRA experts (lora_as/lora_bs) and router (route)
            for name, param in self.backbone.named_parameters():
                if '.lora_as' in name or '.lora_bs' in name or '.route' in name:
                    continue
                param.requires_grad = False

        else:
            raise Exception(transfer_type)

    def set_mode(self, mode):
        assert mode in ['train', 'eval']
        if mode == 'eval':
            self.eval()
        else:
            if self.transfer_type == 'finetune':
                self.train()
            elif self.transfer_type == 'linear':
                self.fc.train()
            elif self.transfer_type == 'mor':
                self.train()
            else:
                raise Exception(self.transfer_type)


