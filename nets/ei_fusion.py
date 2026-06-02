import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, din, dout, hidden_dims, norm=True) -> None:
        super().__init__()
        if norm:
            self.ln = nn.LayerNorm(din, eps=1e-6)
        else:
            self.ln = nn.Identity()
        
        if hidden_dims is None:
            hidden_dims = []
        dims = [din] + hidden_dims + [dout]        

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))   
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                         
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.mlp(x)
        return x


class EIFusion(nn.Module):
    def __init__(self, num_classes, proj_dim,
                 backbone_names, first_backbones, second_backbones,
                 route_sup=None,
                 **kwargs):
        super(EIFusion, self).__init__()
        """
        route_sup: whether to give route supervision signal
        """

        self.num_classes = num_classes

        self.route_sup = route_sup

        self.first_backbones = nn.ModuleDict(first_backbones)
        self.backbone_names = backbone_names
        self.modality_names = list(sorted(self.backbone_names.keys()))

        self.proj_dim = proj_dim

        self.projs_cls = nn.ModuleDict()
        for modality1 in self.backbone_names:
            for modality2 in self.backbone_names:
                if modality1 != modality2:
                    self.projs_cls['{}_{}'.format(modality1, modality2)] = MLP(proj_dim, proj_dim, [384])

        self.second_backbones = nn.ModuleDict(second_backbones)
        for modality in self.second_backbones:
            self.second_backbones[modality].fc = nn.Linear(proj_dim, num_classes)

        self.route = nn.Linear(proj_dim * len(self.modality_names), len(self.modality_names))

    def forward(self, xs):
        # first stage
        first_scores = {}
        first_features = {}
        
        for modality in xs:

            x = xs[modality]
            B, I, C, H, W = x.size()
            first_score, first_feature = self.first_backbones[str(modality)]({modality: x}, return_features=True)  # B * I, num_tokens, embedding_dim
            first_feature_cls = first_feature[:, :1] # B, 1, embedding_dim
            first_features[modality] = {'cls': first_feature_cls}
            first_scores[modality] = first_score
            
        second_scores = {}
        second_features = {}
        # second stage
        for modality in xs:
            prompt_feature = []   
            for _m in first_features:
                if _m == modality:
                    continue
                first_feature_cls = first_features[_m]['cls']
                
                first_feature_cls = self.projs_cls['{}_{}'.format(modality, _m)](first_feature_cls)
                prompt_feature.append(first_feature_cls)
            prompt_feature = torch.cat(prompt_feature, dim=1)  # B, num_prompt_tokens, embedding_dim

            backbone_name = self.backbone_names[str(modality)]
            second_backbone = self.second_backbones[str(modality)]

            x = xs[modality]
            B, I, C, H, W = x.size()
            x = x.view(B * I, C, H, W)
            x = self.forward_pre_blocks(x, second_backbone.backbone, backbone_name)  # B * I, num_tokens, embedding_dim
            x = x.view(B, I, -1, self.proj_dim)  # B, I, num_tokens, embedding_dim
            prompt_feature = prompt_feature.unsqueeze(1).expand(-1, I, -1, -1)

            x = self.forward_blocks(x, second_backbone.backbone, backbone_name, prompt_feature)
            B, I, L, D = x.size()
            x = x.view(B * I, L, D)
            x = self.forward_post_blocks(x, second_backbone.backbone, backbone_name)
            x = x.view(B, I, L, D)  # B, I, num_tokens, embedding_dim
            x = torch.mean(x, dim=1)  # B, num_tokens, embedding_dim
            
            second_features[modality] = {'cls': x[:, :1]}

            x = x[:, 0]  # B, embedding_dim (cls token)
            score = second_backbone.fc(x)
            second_scores[modality] = score

        
        route_feature = []
        scores = []
        for modality in self.modality_names:
            feature = second_features[int(modality)]['cls'].mean(dim=1)   # B, embedding_dim
            route_feature.append(feature)
            scores.append(second_scores[int(modality)])
        
        route_feature = torch.cat(route_feature, dim=1)  # B, embedding_dim * num_modalities
        route_feature = route_feature.detach()
        route_weight = self.route(route_feature)
        route_weight = torch.softmax(route_weight, dim=1)  # B, modalities

        out_scores = torch.stack(scores, dim=1)  # B, modalities, num_classes
        out_scores = out_scores.detach() * route_weight.unsqueeze(-1)
        out_scores = torch.sum(out_scores, dim=1)

        
        if self.route_sup is not None:
            return out_scores, first_scores, second_scores, route_weight
        else:
            return out_scores, first_scores, second_scores
    
    @staticmethod
    def forward_pre_blocks(x, backbone, backbone_name):
        if backbone_name in ['dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g']:
            x = backbone.prepare_tokens_with_masks(x)
        elif backbone_name in ['openclip_vit_b', 'openclip_vit_l']:
            x = backbone.patch_embed(x)
            x = backbone._pos_embed(x)
            x = backbone.patch_drop(x)
            x = backbone.norm_pre(x)
        
        else:
            raise Exception(backbone_name)
          
        return x
            
              
    def forward_blocks(self, x, backbone, backbone_name, prompt):
        if backbone_name in ['dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g', 'openclip_vit_b', 'openclip_vit_l']:
            x = torch.cat([x, prompt], dim=2)  # prepend prompt before the first block
            B, I, L, D = x.size()
            x = x.view(B * I, L, D)
            for blk in backbone.blocks:
                x = blk(x)
            x = x.view(B, I, L, D)
        else:
            raise Exception(backbone_name)

        return x
    
    @staticmethod
    def forward_post_blocks(x, backbone, backbone_name):   
        if backbone_name in ['dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g', 'openclip_vit_b', 'openclip_vit_l']:
            x = backbone.norm(x)
            
        else:
            raise Exception(backbone_name)
        return x  
    
