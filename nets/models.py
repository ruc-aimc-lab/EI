"""Model dispatch for the conference release.

Only four `net` entries are exposed here:
    - dinov2_b           single-modal DINOv2 ViT-B with optional MoR adapter
    - openclip_vit_b     single-modal OpenAI-CLIP ViT-B with optional MoR adapter
    - dinov2_b_ei        multi-modal Early Intervention (EI) over DINOv2-B
    - openclip_vit_b_ei  multi-modal Early Intervention (EI) over OpenCLIP-ViT-B

Both EI variants load pretrained single-modal checkpoints from configs
referenced via `training_params['model_params']['pretrained_configs']` and
`pretrained_ckpts`, then assemble the EI architecture (`EIFusion`).

For single-modal training, the supported `transfer_type` values are:
    - 'finetune'  full fine-tuning
    - 'linear'    linear probing (backbone frozen, only the classifier head trained)
    - 'mor'       Mixture of Low-varied-Ranks Adaptation (the paper's PEFT)
"""
import os
import json
import copy

import torch

from .base_net import BaseNetWithFC
from .backbones import backbones
from .peft.mor import set_dinov2_mor, set_openclip_mor

from .processor import Processor, ProcessorEI
from .ei_fusion import EIFusion


# Lookup order for backbone weights:
#   1. config field (training_params['dinov2_b_ckpt'] / ['openclip_vit_b_ckpt'])
#   2. env var (DINOV2_B_CKPT / OPENCLIP_VIT_B_CKPT)
#   3. hardcoded dev path below
#   4. online download (when checkpoint_path is None; see backbones.create_*)
_DINOV2_DEV_CKPT = '/data2/wqj/checkpoints/dinov2_vitb14_reg4_pretrain.pth'
_OPENCLIP_DEV_CKPT = '/data2/wqj/checkpoints/vit_base_patch16_clip_224.openai'


def _resolve_ckpt(config_value, env_var, dev_path):
    """Return the first existing path among (config, env, dev); None triggers online."""
    for candidate in (config_value, os.environ.get(env_var), dev_path):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def build_model(model_name, training_params, training, dataset_type, run_num):
    n_class = training_params['n_class']
    custom_pretrained = training_params['custom_pretrained']
    model = getattr(Models, model_name)(
        n_class=n_class, custom_pretrained=custom_pretrained,
        dataset_type=dataset_type, training_params=training_params,
        training=training, run_num=run_num,
    )
    return model


def _apply_mor(backbone, training_params, set_mor_fn):
    lora_params = training_params['lora_params']
    r = lora_params['r']
    alpha = lora_params['alpha']
    lora_layers = lora_params['lora_layers']
    num_loras = lora_params['num_loras']
    print('mor: r={} alpha={} layers={} num_loras={}'.format(
        r, alpha, lora_layers, num_loras))
    set_mor_fn(backbone, r=r, alpha=alpha, lora_layers=lora_layers,
               num_loras=num_loras)


class Models(object):

    @staticmethod
    def dinov2_b(n_class, custom_pretrained, dataset_type, training_params, training, **kwargs):
        transfer_type = training_params['transfer_type']

        embedding_dim = 768
        backbone_name = 'dinov2_b'

        checkpoint_path = _resolve_ckpt(
            training_params.get('dinov2_b_ckpt'), 'DINOV2_B_CKPT', _DINOV2_DEV_CKPT,
        )
        backbone = backbones.create_dinov2(checkpoint_path=checkpoint_path)

        if custom_pretrained:
            checkpoint = torch.load(custom_pretrained, map_location='cpu')
            msg = backbone.load_state_dict(checkpoint, strict=False)
            print('checkpoint missing_keys', msg.missing_keys)

        if transfer_type == 'mor':
            _apply_mor(backbone, training_params, set_dinov2_mor)
        elif transfer_type not in ['finetune', 'linear']:
            raise Exception('Unsupported transfer_type: {}'.format(transfer_type))

        model = BaseNetWithFC(
            backbone=backbone, embedding_dim=embedding_dim, n_class=n_class,
            backbone_name=backbone_name,
        )
        model.set_grad(transfer_type=transfer_type)

        if dataset_type not in ['derm', 'mmc', 'mrnet']:
            raise Exception(dataset_type)
        return Processor(model=model, training_params=training_params, training=training)

    @staticmethod
    def openclip_vit_b(n_class, custom_pretrained, dataset_type, training_params, training, **kwargs):
        transfer_type = training_params['transfer_type']

        embedding_dim = 768
        backbone_name = 'openclip_vit_b'
        img_size = training_params['img_size']

        checkpoint_path = _resolve_ckpt(
            training_params.get('openclip_vit_b_ckpt'), 'OPENCLIP_VIT_B_CKPT', _OPENCLIP_DEV_CKPT,
        )
        backbone = backbones.create_openclip_vit_b(checkpoint_path=checkpoint_path, img_size=img_size)

        if custom_pretrained:
            checkpoint = torch.load(custom_pretrained, map_location='cpu')
            msg = backbone.load_state_dict(checkpoint, strict=False)
            print('checkpoint missing_keys', msg.missing_keys)

        if transfer_type == 'mor':
            _apply_mor(backbone, training_params, set_openclip_mor)
        elif transfer_type not in ['finetune', 'linear']:
            raise Exception('Unsupported transfer_type: {}'.format(transfer_type))

        model = BaseNetWithFC(
            backbone=backbone, embedding_dim=embedding_dim, n_class=n_class,
            backbone_name=backbone_name,
        )
        model.set_grad(transfer_type=transfer_type)

        if dataset_type not in ['derm', 'mmc', 'mrnet']:
            raise Exception(dataset_type)
        return Processor(model=model, training_params=training_params, training=training)

    @staticmethod
    def dinov2_b_ei(n_class, custom_pretrained, dataset_type, training_params, training, **kwargs):
        return _build_ei(
            n_class=n_class, dataset_type=dataset_type,
            training_params=training_params, training=training,
            backbone_name='dinov2_b', embedding_dim=768, run_num=kwargs['run_num'],
        )

    @staticmethod
    def openclip_vit_b_ei(n_class, custom_pretrained, dataset_type, training_params, training, **kwargs):
        return _build_ei(
            n_class=n_class, dataset_type=dataset_type,
            training_params=training_params, training=training,
            backbone_name='openclip_vit_b', embedding_dim=768, run_num=kwargs['run_num'],
        )


def _build_ei(n_class, dataset_type, training_params, training,
                 backbone_name, embedding_dim, run_num):
    """Build the EI multi-modal model from pretrained single-modal backbones."""
    modality_idx = training_params['modality_idx']
    model_params = training_params['model_params']

    pretrained_configs = model_params['pretrained_configs']
    pretrained_ckpts = list(model_params['pretrained_ckpts'])  # copy: we rewrite paths in-place
    run_num = run_num % 3

    for i in range(len(pretrained_ckpts)):
        if pretrained_ckpts[i] is not None:
            pretrained_ckpts[i] = os.path.join(pretrained_ckpts[i], 'runs_{}'.format(run_num), 'best_model.pkl')

    first_backbones = {}
    backbone_names = {}
    for pretrained_config, pretrained_ckpt, modality in zip(pretrained_configs, pretrained_ckpts, modality_idx):
        with open(pretrained_config) as fin:
            pre_config = json.load(fin)
            pre_training_params = pre_config['training_params']
            pre_model_name = pre_training_params['net']
            pre_model = getattr(Models, pre_model_name)(
                n_class=n_class, custom_pretrained=None,
                dataset_type=dataset_type, training_params=pre_training_params,
                training=False, run_num=run_num,
            )
            if pretrained_ckpt is not None:
                pre_model.load_model(pretrained_ckpt)
            pre_model = pre_model.model
            first_backbones[str(modality)] = pre_model
            backbone_names[str(modality)] = backbone_name

    route_sup = training_params.get('route_sup', None)

    second_backbones = {}
    for pretrained_config, pretrained_ckpt, modality in zip(pretrained_configs, pretrained_ckpts, modality_idx):
        with open(pretrained_config) as fin:
            pre_config = json.load(fin)
            pre_training_params = pre_config['training_params']
            pre_model_name = pre_training_params['net']
            pre_model = getattr(Models, pre_model_name)(
                n_class=n_class, custom_pretrained=None,
                dataset_type=dataset_type, training_params=pre_training_params,
                training=False, run_num=run_num,
            )
            if pretrained_ckpt is not None:
                pre_model.load_model(pretrained_ckpt)
            pre_model = pre_model.model
            second_backbones[str(modality)] = pre_model

    model = EIFusion(
        num_classes=n_class, first_backbones=first_backbones,
        second_backbones=second_backbones, backbone_names=backbone_names,
        proj_dim=embedding_dim,
        route_sup=route_sup,
    )

    if dataset_type not in ['derm', 'mmc', 'mrnet']:
        raise Exception(dataset_type)
    return ProcessorEI(model=model, training_params=training_params, training=training)
