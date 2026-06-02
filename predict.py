import os
import sys
import json

from nets import build_model
from data import build_dataloader
import numpy as np
from predictor import Predictor


def main(train_collection, val_collection, config_path, test_collection, run_num, device):
    # device='-1' means accelerate was used for training; inference runs on cuda:0
    if device == '-1':
        device = [0]
    else:
        device = list(map(int, device.split(',')))
    with open(config_path, 'r') as fin:
        config = json.load(fin)
    config_name = config_path.split(os.sep)[-1]

    training_params = config['training_params']
    paths = config['paths']
    augmentation_params = config['augmentation_params']

    output_root = paths['output_root']
    mapping_path = paths['mapping_path']
    dataset_type = training_params['dataset_type']
    
    test_loader = build_dataloader(
        paths=paths, training_params=training_params, 
        augmentation_params=augmentation_params, 
        collection_name=test_collection, 
        mapping_path=mapping_path, train=False)
    
    inter_val = int(test_loader.dataset.__len__() / test_loader.batch_size) + 1
    training_params['inter_val'] = inter_val
    
    run_num = int(run_num)
    model = build_model(training_params['net'], training_params, 
                        training=False,  dataset_type=dataset_type, run_num=run_num)
    model.requires_grad_false()
    model.set_mode('eval')

    model_path = os.path.join(output_root, train_collection, 'Models', val_collection, config_name, 'runs_{}'.format(run_num), 'best_model.pkl')
    model.load_model(model_path)
    model.set_device(device)

    out_root = os.path.join(output_root, test_collection, 'Predictions', train_collection,
                            val_collection, config_name, 'runs_{}'.format(run_num))
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    #else:
    #    raise Exception('prediction \n{}\nalready exists'.format(out_root))

    predictor = Predictor(model, test_loader)

    img_paths, scores, targets = predictor.predict(device)
    # Apply sigmoid uniformly across datasets. For multi-label (MRNet) this is the
    # correct activation. For multi-class (Derm7pt, MMC-AMD) softmax would be the
    # "natural" choice, but every downstream metric (per-class AP/AUC and argmax-
    # based precision/recall/F1/accuracy) is invariant under sigmoid's monotonic
    # transform — so the reported numbers are identical either way.
    scores = 1.0 / (1.0 + np.exp(-scores))
    
    out_name = 'results.csv'
    with open(os.path.join(out_root, out_name), 'w') as fout:
        for img_path, score in zip(img_paths, scores):
            score = ','.join(list(map(lambda x: '{:.4f}'.format(x), score)))
            line = '{}\t{}\n'.format(img_path, score)
            fout.write(line)
    
    
if __name__ == '__main__':
    train_collection = sys.argv[1]
    val_collection = sys.argv[2]
    config_path = sys.argv[3]
    test_collection = sys.argv[4]
    run_num = sys.argv[5]
    device = sys.argv[6]

    main(train_collection, val_collection, config_path, test_collection, run_num, device)
   