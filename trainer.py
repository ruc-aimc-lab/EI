# model training
import os
import time
import torch

from utils import Evaluater
from data import build_dataloader
from predictor import Predictor
import numpy as np

from nets import build_model

try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


class Trainer(object):
    def __init__(self, paths, training_params, augmentation_params, device):

        self.use_accelerator = (device == '-1')
        if self.use_accelerator:
            if not ACCELERATE_AVAILABLE:
                raise ImportError('accelerate is not installed. Run: pip install accelerate')
            from accelerate import DistributedDataParallelKwargs
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            # construct Accelerator up front so run_num can be broadcast across
            # ranks before anyone scans the output directory (otherwise ranks
            # race on os.path.exists and pick different run_num values).
            self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
            self.device = [-1]
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            self._original_batch_size = training_params['batch_size']
            training_params['batch_size'] = max(1, training_params['batch_size'] // world_size)
        else:
            self.accelerator = None
            self.device = list(map(int, device.split(',')))
        self.training_params = training_params
        self.paths = paths
        self.augmentation_params = augmentation_params

        self.imroot = paths['image_root']

        self.train_collection = paths['train_collection']
        self.val_collection = paths['val_collection']

        self.output_root = paths['output_root']
        self.config_path = paths['config_path']
        self.config_name = self.config_path.split(os.sep)[-1]

        self.mapping_path = paths['mapping_path']
        
        self.dataset_type = training_params['dataset_type']
        self.num_workers = self.training_params['num_workers']

        self.evaluater = Evaluater()

        # dataloaders
        self.train_loader = build_dataloader(
            paths=self.paths, training_params=self.training_params, 
            augmentation_params=self.augmentation_params, 
            collection_name=self.train_collection, 
            mapping_path=self.mapping_path, train=True)
        # val uses original batch_size (runs on rank 0 only, no need to split)
        if self.use_accelerator:
            training_params['batch_size'] = self._original_batch_size
        self.val_loader = build_dataloader(
            paths=self.paths, training_params=self.training_params,
            augmentation_params=self.augmentation_params,
            collection_name=self.val_collection,
            mapping_path=self.mapping_path, train=False)
        if self.use_accelerator:
            training_params['batch_size'] = max(1, self._original_batch_size // world_size)
        
        if self.use_accelerator:
            # Accelerate distributes data: each rank sees 1/world_size of samples
            self.inter_val = int(self.train_loader.dataset.__len__() / (self.train_loader.batch_size * world_size)) + 1
        else:
            self.inter_val = int(self.train_loader.dataset.__len__() / self.train_loader.batch_size) + 1

        self.training_params['inter_val'] = self.inter_val
        
        # output directory -- main rank chooses run_num by scanning existing
        # runs_<n>/ and creates the new dir; other ranks receive the same
        # value via broadcast so they cannot race on os.path.exists (a race
        # here would make EI load mismatched pretrained checkpoints).
        models_root = os.path.join(self.output_root, self.train_collection, 'Models', self.val_collection, self.config_name)
        if self.accelerator is None or self.accelerator.is_main_process:
            self.run_num = 0
            self.out = os.path.join(models_root, 'runs_{}'.format(self.run_num))
            while os.path.exists(self.out):
                self.run_num += 1
                self.out = os.path.join(models_root, 'runs_{}'.format(self.run_num))
            os.makedirs(self.out, exist_ok=True)
        else:
            self.run_num = 0
        if self.use_accelerator:
            from accelerate.utils import broadcast_object_list
            run_num_list = [self.run_num]
            broadcast_object_list(run_num_list, from_process=0)
            self.run_num = run_num_list[0]
            self.out = os.path.join(models_root, 'runs_{}'.format(self.run_num))
            self.accelerator.wait_for_everyone()
          
        self.model = build_model(model_name=training_params['net'], training_params=training_params, 
                                 training=True, dataset_type=self.dataset_type, run_num=self.run_num)
        
        if self.use_accelerator:
            self._setup_accelerator()
        else:
            self.model.set_device(self.device)
        self.model.set_mode('train')

        print('finish model loading')
        
        # output dir was already created on the main rank during run_num assignment
        is_main = not self.use_accelerator or self.accelerator.is_main_process

        self.log_headers = ['iteration', 'train/loss',
                            'train/sensitivity', 'train/specificity', 'train/f1', 'train/auc', 'train/ap', 'train/acc',
                            'valid/sensitivity', 'valid/specificity', 'valid/f1', 'valid/auc', 'valid/ap', 'valid/acc',
                            'total_time']

        # main rank keeps log.csv open across the whole run (line-buffered)
        if is_main:
            self.log_file = open(os.path.join(self.out, 'log.csv'), 'w', buffering=1)
            self.log_file.write(','.join(self.log_headers) + '\n')
        else:
            self.log_file = None
        
        self.iteration = 0
        self.metric = -1
        self.no_improve = 0
        self.start_time = time.time()
        self.end = False

        print('model: {}'.format(training_params['net']))
        print('dataset: ', self.train_collection, self.val_collection)

    def _setup_accelerator(self):
        # Accelerator was constructed in __init__ so run_num could be broadcast
        # before this point; here we only wrap the model / optimizer / loader.
        # NOTE: deliberately do NOT prepare the lr scheduler. AcceleratedScheduler
        # steps the underlying scheduler num_processes times per .step() call,
        # assuming the dataloader batch was multiplied by num_processes. This
        # codebase does the opposite -- it divides batch_size by world_size to
        # keep the global batch (and therefore the per-epoch iteration count /
        # inter_val) identical to single-GPU. Wrapping the scheduler would make
        # CyclicLR cycle num_processes x too fast and significantly degrade
        # accuracy. The raw torch scheduler shares param_groups with the prepared
        # optimizer, so stepping it manually keeps the LR trajectory aligned with
        # the single-GPU run.
        prepared_model, prepared_optim, prepared_loader = self.accelerator.prepare(
            self.model.model, self.model.opt.optim, self.train_loader
        )
        self.model.model = prepared_model
        self.model.opt.optim = prepared_optim
        self.train_loader = prepared_loader
        self.model.accelerator = self.accelerator
        print('accelerate enabled, device:', self.accelerator.device,
              'num_processes:', self.accelerator.num_processes,
              'mixed_precision:', self.accelerator.mixed_precision,
              'split_batches:', self.accelerator.split_batches)
        for batch in self.train_loader:
            _, imgs, target = batch
            first_key = list(imgs.keys())[0]
            print(f'[rank {self.accelerator.process_index}] actual batch_size={target.shape[0]}, '
                  f'img_shape={imgs[first_key].shape}')
            break

    def _is_main_process(self):
        if self.use_accelerator:
            return self.accelerator.is_main_process
        return True

    def validate(self):
        if self.use_accelerator and not self.accelerator.is_main_process:
            # non-main ranks wait for main to finish validation
            self.accelerator.wait_for_everyone()
            return

        print('validating...')
        # use unwrapped model for single-rank validation
        if self.use_accelerator:
            raw_model = self.accelerator.unwrap_model(self.model.model)
            original_model = self.model.model
            self.model.model = raw_model

        predictor = Predictor(self.model, self.val_loader)
        _, scores, targets = predictor.predict(device=self.device)

        if self.use_accelerator:
            self.model.model = original_model

        hist, sensitivity, specificity, f1, auc, ap, acc = self.evaluater.evaluate(scores, targets)
        sensitivity, specificity, f1, auc, ap, acc = np.nanmean(sensitivity), np.nanmean(specificity), np.nanmean(f1), np.nanmean(auc), np.nanmean(ap), np.nanmean(acc)

        if self._is_main_process():
            log_iter = [self.iteration, '']
            log_train = [''] * 6
            log_test = [sensitivity, specificity, f1, auc, ap, acc]
            total_time = time.time() - self.start_time

            log_iter = ','.join(list(map(str, log_iter)))
            log_train = ','.join(log_train)
            log_test = ','.join(list(map(lambda x: '{:.4f}'.format(x), log_test)))

            log = '{},{},{},{:.2f}\n'.format(log_iter, log_train, log_test, total_time)
            self.log_file.write(log)

        metric = ap

        is_best = metric > self.metric
        if is_best:
            self.metric = metric
            self.no_improve = 0
            self.model.save_model(os.path.join(self.out, 'best_model.pkl'))
            print('model saved')
        else:
            self.no_improve += 1

        if self.no_improve >= 10:
            self.end = True

        if self.use_accelerator:
            self.accelerator.wait_for_everyone()

    def train(self):
        for epoch in range(200):
            if self.end:
                break
            for data_in in self.train_loader:
                if self.iteration % self.inter_val  == 0:
                    self.model.set_mode('eval')
                    self.validate()
                    # sync early stopping flag across all ranks
                    if self.use_accelerator:
                        end_tensor = torch.tensor([1 if self.end else 0],
                                                  device=self.accelerator.device)
                        end_tensor = self.accelerator.reduce(end_tensor, reduction='max')
                        self.end = end_tensor.item() > 0
                    self.model.set_mode('train')
                    if self.end:
                        break
                self.iteration += 1
                
                img_path, imgs, target = data_in
                score, train_loss = self.model.fit(xs=imgs, ys=target, device=self.device)

                if self.use_accelerator and self.accelerator.num_processes > 1:
                    score = self.accelerator.gather(score.detach()).cpu().numpy()
                    target = self.accelerator.gather(target.to(self.accelerator.device)).cpu().numpy()
                else:
                    score = score.data.cpu().numpy()
                    target = target.data.cpu().numpy()
                
                hist, sensitivity, specificity, f1, auc, ap, acc = self.evaluater.evaluate(score, target)
                sensitivity, specificity, f1, auc, ap, acc = np.nanmean(sensitivity), np.nanmean(specificity), np.nanmean(f1), np.nanmean(auc), np.nanmean(ap), np.nanmean(acc)

                metric = ap

                total_time = time.time() - self.start_time
                if self._is_main_process():
                    print('iteration {:d}, loss={:.3f}, lr={:.3e}, metric={:.3f}, max_metric={:.3f}, no_improve:{:d}'.format(
                        self.iteration, train_loss.data.item(), self.model.opt.get_lr(), metric, self.metric, self.no_improve))

                    log_iter = [self.iteration, train_loss.data.item()]
                    log_train = [sensitivity, specificity, f1, auc, ap, acc]
                    log_test = [''] * 6

                    log_iter = ','.join(list(map(str, log_iter)))
                    log_test = ','.join(log_test)
                    log_train = ','.join(list(map(lambda x: '{:.4f}'.format(x), log_train)))

                    log = '{},{},{},{:.2f}\n'.format(log_iter, log_train, log_test, total_time)
                    self.log_file.write(log)

        if self.log_file is not None:
            self.log_file.close()
