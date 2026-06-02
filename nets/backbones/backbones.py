import os

import torch
import torch.nn as nn
import timm


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


_OPENCLIP_TIMM_NAME = 'vit_base_patch16_clip_224.openai'
_DINOV2_HUB_DIR = '/data2/wqj/dinov2'              # local clone of facebookresearch/dinov2
_DINOV2_HUB_REPO = 'facebookresearch/dinov2'
_DINOV2_MODEL_NAME = 'dinov2_vitb14_reg'


def create_openclip_vit_b(checkpoint_path, img_size):
    """Load OpenAI-CLIP ViT-B/16, then adapt patch embed / pos embed to `img_size`.

    Local: `checkpoint_path` is a folder containing `model.pth`.
    Online fallback: kicks in when the local file is missing.
    """
    if checkpoint_path and os.path.exists(os.path.join(checkpoint_path, 'model.pth')):
        backbone = timm.create_model(_OPENCLIP_TIMM_NAME, pretrained=False)
        checkpoint = torch.load(os.path.join(checkpoint_path, 'model.pth'), map_location='cpu')
    else:
        backbone = timm.create_model(_OPENCLIP_TIMM_NAME, pretrained=True)
        checkpoint = backbone.state_dict()  # capture before mutating patch_embed/pos_embed

    backbone.patch_embed = timm.layers.PatchEmbed(
        img_size=(img_size, img_size), patch_size=16, in_chans=3,
        embed_dim=768, bias=False, dynamic_img_pad=False,
    )
    backbone.pos_embed = nn.Parameter(
        torch.randn(1, backbone.patch_embed.num_patches + backbone.num_prefix_tokens, 768) * .02
    )
    interpolate_pos_embed(backbone, checkpoint)

    backbone.load_state_dict(checkpoint)
    backbone.reset_classifier(-1, None)
    return backbone


def create_dinov2(checkpoint_path):
    """Load DINOv2 ViT-B/14 (with register tokens).

    Local: `checkpoint_path` is the `.pth` file; the architecture is loaded
    via `source='local'` from `_DINOV2_HUB_DIR` (a local clone of
    facebookresearch/dinov2), and `model_name` is derived from the filename
    (e.g. `dinov2_vitb14_reg4_pretrain.pth` -> `dinov2_vitb14_reg`).
    Online fallback: kicks in when the local file is missing.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        model_name = checkpoint_path.split(os.sep)[-1]
        model_name = '_'.join(model_name.split('_')[:-1])[:-1]
        backbone = torch.hub.load(_DINOV2_HUB_DIR, model_name, source='local', pretrained=False)
        backbone.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    else:
        backbone = torch.hub.load(_DINOV2_HUB_REPO, _DINOV2_MODEL_NAME, pretrained=True)
    return backbone
