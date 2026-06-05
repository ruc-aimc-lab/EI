import math
import torch.optim as optim
import torch.nn as nn
import torch


def _get_layer_id_for_vit(name, num_layers):
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def _param_groups_lrd(model, weight_decay, no_weight_decay_list, layer_decay):
    """Layer-wise LR decay for ViT, following BEiT/ELECTRA."""
    param_groups = {}
    num_layers = len(model.blocks) + 1
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = _get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            param_groups[group_name] = {
                "lr_scale": layer_scales[layer_id],
                "weight_decay": this_decay,
                "params": [],
            }
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


class _CosineWithWarmup(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min_ratio, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            # clamp progress so that once step exceeds total_steps the LR stays
            # at eta_min instead of cycling back up toward the peak.
            progress = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_val
        super().__init__(optimizer, lr_lambda, last_epoch)


class Optimizer(object):
    def __init__(self, models, training_params):
        self.lr = training_params['lr']
        self.weight_decay = training_params['weight_decay']
        method = training_params['optimizer']
        layer_decay = training_params.get('layer_decay', None)

        if method == 'SGD':
            params = []
            for model in models:
                if isinstance(model, nn.Parameter):
                    params += [model]
                else:
                    params += list(model.parameters())
            self.momentum = training_params['momentum']
            self.optim = optim.SGD(params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        elif method == 'ADAMW':
            betas = training_params.get('betas', (0.9, 0.999))

            if (layer_decay is not None
                    and len(models) == 1
                    and hasattr(models[0], 'backbone')
                    and hasattr(models[0].backbone, 'blocks')):
                backbone = models[0].backbone
                no_wd_list = backbone.no_weight_decay() if hasattr(backbone, 'no_weight_decay') else set()
                backbone_groups = _param_groups_lrd(backbone, self.weight_decay, no_wd_list, layer_decay)
                # Set per-group lr: base_lr * lr_scale
                for g in backbone_groups:
                    g['lr'] = self.lr * g['lr_scale']
                # Head and other non-backbone params at full lr
                backbone_param_ids = {id(p) for p in backbone.parameters()}
                head_params = [p for m in models for p in m.parameters()
                               if p.requires_grad and id(p) not in backbone_param_ids]
                if head_params:
                    backbone_groups.append({
                        'params': head_params,
                        'weight_decay': self.weight_decay,
                        'lr': self.lr,
                    })
                param_groups = backbone_groups
            else:
                params = []
                for model in models:
                    if isinstance(model, nn.Parameter):
                        params += [model]
                    else:
                        params += list(model.parameters())
                param_groups = params

            self.optim = optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay, betas=tuple(betas))

        else:
            raise Exception('')

        schedule_name = training_params['lr_schedule']
        schedule_params = training_params['schedule_params']

        if schedule_name == 'CosineAnnealingLR':
            schedule_params['T_max'] = training_params['inter_val'] * 4
        elif schedule_name == 'CyclicLR':
            scale_fn = schedule_params.get('scale_fn', None)
            if scale_fn is not None:
                schedule_params['scale_fn'] = lambda epoch: scale_fn ** (epoch - 1)
            schedule_params['step_size_up'] = training_params['inter_val'] * 4
        elif schedule_name == 'CosineWithWarmup':
            inter_val = training_params['inter_val']
            warmup_steps = int(schedule_params['warmup_epochs'] * inter_val)
            total_steps = int(schedule_params['total_epochs'] * inter_val)
            eta_min = schedule_params.get('eta_min', 1e-6)
            eta_min_ratio = eta_min / self.lr
            self.lr_schedule = _CosineWithWarmup(self.optim, warmup_steps, total_steps, eta_min_ratio)
            return

        self.lr_schedule = getattr(optim.lr_scheduler, schedule_name)(self.optim, **schedule_params)

    def update_lr(self):
        self.lr_schedule.step()

    def z_grad(self):
        self.optim.zero_grad()

    def g_step(self):
        self.optim.step()

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']
