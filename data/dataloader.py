import os
from .dataset import derm7ptDataset, mmc_amdDataset, mrnetDataset
from torch.utils.data import DataLoader
from .augmentation import OurAug
import json

def build_dataloader(paths, training_params, augmentation_params, collection_name, mapping_path, train, only_label=False):
    lstpaths = []
    for collection in collection_name.split('+'):
        lstpath = os.path.join(paths['collection_root'], collection, 'anno.txt')
        lstpaths.append(lstpath)
    im_root = paths['image_root']
    
    dataset_type = training_params['dataset_type']
    num_class = training_params['n_class']
    batch_size = training_params['batch_size']
    num_workers = training_params['num_workers']
    modality_idx = training_params['modality_idx']
    
    augs = {}
    if train:
        for idx in modality_idx:
            aug = OurAug(augmentation_params[idx])
            augs[idx] = aug
    else:
        for idx in modality_idx:
            aug = OurAug({'size_h': augmentation_params[idx]['size_h'], 
                          'size_w': augmentation_params[idx]['size_w']})
            augs[idx] = aug

    with open(mapping_path, 'r') as fin:
        mappings = fin.read()
        mappings = json.loads(mappings)

    if dataset_type == 'derm':
        dataset = derm7ptDataset(
            lstpath=lstpaths,
            img_root=im_root,
            num_class=num_class,
            mappings=mappings,
            modality_idx=modality_idx,
            augs=augs,
            train=train,
            only_label=only_label
            )
    elif dataset_type == 'mmc':
        dataset = mmc_amdDataset(
            lstpath=lstpaths,
            img_root=im_root,
            num_class=num_class,
            mappings=mappings,
            modality_idx=modality_idx,
            augs=augs,
            train=train,
            only_label=only_label
            )
    elif dataset_type == 'mrnet':
        sampling_num = training_params['sampling_num']
        dataset = mrnetDataset(
            lstpath=lstpaths,
            img_root=im_root,
            num_class=num_class,
            mappings=mappings,
            modality_idx=modality_idx,
            augs=augs,
            train=train,
            only_label=only_label,
            sampling_num=sampling_num,
        )
    
    else:
        raise Exception(dataset_type)
    
    dataloader = DataLoader(dataset, shuffle=train, batch_size=batch_size, num_workers=num_workers)

    return dataloader

