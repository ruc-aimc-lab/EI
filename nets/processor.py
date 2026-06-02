import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .optimizer import Optimizer


class BasicProcessor(object):
    def __init__(self, model, training_params, training=True):
        self.model = model
        self.accelerator = None  # set externally when using accelerate

        self.loss_func = training_params['loss_func']
        self.loss_params = training_params['loss_params']

        if training:
            self.opt = Optimizer([self.model], training_params)

            if 'weight' in self.loss_params:
                self.loss_params['weight'] = torch.FloatTensor(self.loss_params['weight'])

            if self.loss_func == 'bce':
                self.crit = nn.BCEWithLogitsLoss(**self.loss_params)
                self.target_dtype = torch.FloatTensor
            elif self.loss_func == 'ce':
                self.crit = nn.CrossEntropyLoss(**self.loss_params)
                self.target_dtype = torch.LongTensor
            else:
                raise Exception('Unknown loss function {}'.format(self.loss_func))

    def _get_device(self, device):
        if self.accelerator is not None:
            return self.accelerator.device
        if len(device) > 1:
            return 'cuda'
        return 'cuda:{}'.format(device[0])

    def _backward(self, loss):
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()

    def requires_grad_false(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_mode(self, mode):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise Exception('Invalid model mode {}'.format(mode))

    def set_device(self, device):
        if self.accelerator is not None:
            return
        print('set_device', device)
        if len(device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device)
            _device = 'cuda'
        else:
            _device = 'cuda:{}'.format(device[0])
        self.model.to(_device)

    def save_model(self, path):
        if self.accelerator is not None:
            unwrapped = self.accelerator.unwrap_model(self.model)
            self.accelerator.save(unwrapped.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location='cpu')

        remove_module = True
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                remove_module = False
                break
        if remove_module:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v

            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)


class Processor(BasicProcessor):
    """Single-modal processor for both finetune/linear/MoR adaptation."""

    def __init__(self, model, training_params, training=True) -> None:
        super().__init__(model, training_params, training)

    def fit(self, xs, ys, device, **kwargs):
        self.opt.z_grad()

        _device = self._get_device(device)

        for modality in xs:
            xs[modality] = xs[modality].type(torch.FloatTensor).to(_device)

        ys = ys.type(self.target_dtype).to(_device)

        scores = self.model(xs)
        loss = self.crit(scores, ys)

        self._backward(loss)

        self.opt.g_step()
        self.opt.update_lr()

        return scores, loss

    def predict(self, xs, device):
        _device = self._get_device(device)
        for modality in xs:
            if xs[modality] is not None:
                xs[modality] = xs[modality].type(torch.FloatTensor).to(_device)

        with torch.inference_mode():
            scores = self.model(xs)
        return scores


class ProcessorEI(BasicProcessor):
    """Multi-modal processor for Early Intervention (EI) fusion.

    Combines a multi-modal classification loss with two auxiliary single-modal
    losses (first-round and second-round predictions per modality), and an
    optional router supervision when route_sup is enabled.
    """

    def __init__(self, model, training_params, training=True) -> None:
        super().__init__(model, training_params, training)
        self.mm_loss_weight = training_params['mm_loss_weight']
        self.first_single_loss_weight = training_params['first_single_loss_weight']
        self.second_single_loss_weight = training_params['second_single_loss_weight']
        self.modality_names = list(sorted(self.model.backbone_names.keys()))

        self.route_sup = training_params.get('route_sup', None)
        if self.route_sup is not None:
            if self.route_sup == 'ce':
                self.route_sup_loss_weight = training_params['route_sup_loss_weight']
                self.prior_gaps = training_params['prior_gaps']
                self.prior_gaps = torch.tensor(self.prior_gaps).unsqueeze(1)
                self.score_sup = training_params.get('score_sup', False)

    def fit(self, xs, ys, device, **kwargs):
        self.opt.z_grad()

        _device = self._get_device(device)

        for modality in xs:
            xs[modality] = xs[modality].type(torch.FloatTensor).to(_device)

        ys = ys.type(self.target_dtype).to(_device)
        if self.route_sup is not None:
            scores, first_scores, second_scores, route_weight = self.model(xs)

            if self.route_sup == 'ce':
                modality_performance = []
                for modality in self.modality_names:
                    p = second_scores[int(modality)]
                    perf = F.cross_entropy(p, ys, reduction='none')
                    modality_performance.append(perf)
                modality_performance = torch.stack(modality_performance, dim=0)
                modality_performance *= self.prior_gaps.to(_device)
                route_target = torch.argmin(modality_performance, dim=0).detach()
            elif self.route_sup == 'decision':
                pass

            loss = F.cross_entropy(route_weight, route_target) * self.route_sup_loss_weight
            if self.score_sup:
                loss += self.crit(scores, ys) * self.mm_loss_weight

            if self.first_single_loss_weight > 0:
                for modality in first_scores:
                    first_loss = self.crit(first_scores[modality], ys)
                    loss += self.first_single_loss_weight * first_loss
            if self.second_single_loss_weight > 0:
                for modality in second_scores:
                    second_loss = self.crit(second_scores[modality], ys)
                    loss += self.second_single_loss_weight * second_loss
        else:
            scores, first_scores, second_scores = self.model(xs)
            loss = self.crit(scores, ys) * self.mm_loss_weight

            if self.first_single_loss_weight > 0:
                for modality in first_scores:
                    first_loss = self.crit(first_scores[modality], ys)
                    loss += self.first_single_loss_weight * first_loss
            if self.second_single_loss_weight > 0:
                for modality in second_scores:
                    second_loss = self.crit(second_scores[modality], ys)
                    loss += self.second_single_loss_weight * second_loss

        self._backward(loss)

        self.opt.g_step()
        self.opt.update_lr()

        return scores, loss

    def predict(self, xs, device):
        _device = self._get_device(device)
        for modality in xs:
            xs[modality] = xs[modality].type(torch.FloatTensor).to(_device)

        with torch.inference_mode():
            if self.route_sup is not None:
                scores, first_scores, second_scores, route_weight = self.model(xs)
            else:
                scores, first_scores, second_scores = self.model(xs)
        return scores
