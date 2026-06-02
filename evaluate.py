import sys
import os
import json
import numpy as np
from utils import Evaluater
from data import build_dataloader

def load_pred(pred_path):
    with open(pred_path) as fin:
        lines = fin.readlines()
    pred_dic = {}
    for line in lines:
        line = line.strip('\r\n').split('\t')
        im_name = line[0]
        score = line[1].split(',')
        score = list(map(float, score))
        pred_dic[im_name] = score
    return pred_dic


def main(train_collection, val_collection, config_path, test_collection, run_num):
    with open(config_path, 'r') as fin:
        config = json.load(fin)
    config_name = config_path.split(os.sep)[-1]

    training_params = config['training_params']
    paths = config['paths']
    augmentation_params = config['augmentation_params']
    output_root = paths['output_root']

    out_name = 'results.csv'

    pred_path = os.path.join(output_root, test_collection, 'Predictions', train_collection,
                            val_collection, config_name, 'runs_{}'.format(run_num), out_name)

    
    mapping_path = paths['mapping_path']
    test_loader = build_dataloader(
        paths=paths, training_params=training_params, 
        augmentation_params=augmentation_params, 
        collection_name=test_collection, 
        mapping_path=mapping_path, train=False, only_label=True)
    
    gt_dic = {}
    for data_in in test_loader:
        img_paths = data_in[0]
        gts = data_in[-1]
        gts = gts.data.cpu().numpy()
        for img_path, gt in zip(img_paths, gts):
            gt_dic[img_path] = gt   
    pred_dic = load_pred(pred_path)
    
    with open(mapping_path, 'r') as fin:
        mappings = fin.read()
        mappings = json.loads(mappings)
        reserve_mappings = {}
        for k in mappings:
            reserve_mappings[mappings[k]] = k
    gts = []
    preds = []
    for im_name in gt_dic:
        gts.append(gt_dic[im_name])
        preds.append(pred_dic[im_name])
    gts = np.array(gts)
    preds = np.array(preds)
    evaluater = Evaluater()
    
    hist, sensitivity, specificity, f1, auc, ap, acc = evaluater.evaluate(preds, gts, thre=0.5)
        
    row_heads = ['sensitivity', 'specificity', 'f1', 'auc', 'ap', 'accuracy']
    column_head = ['mean'] + [reserve_mappings[i] for i in range(len(sensitivity))]
    results = [sensitivity, specificity, f1, auc, ap, acc]

    eval_out_name = 'eval_results.csv'
    
    with open(os.path.join(output_root, test_collection, 'Predictions', train_collection,
                            val_collection, config_name, 'runs_{}'.format(run_num), eval_out_name), 'w') as fout:
        column_head = ','.join(column_head)
        fout.write(',{}\n'.format(column_head))

        for i in range(len(row_heads)):
            row_head = row_heads[i]
            result = results[i]
            result_mean = np.nanmean(result)
            if isinstance(result, np.float64): 
                result = ''
            else:
                result = ','.join(list(map(lambda x: '{:.4f}'.format(x), result)))
            fout.write('{},{:.4f},{}\n'.format(row_head, result_mean, result))
    

if __name__ == '__main__':
    train_collection = sys.argv[1]
    val_collection = sys.argv[2]
    config_path = sys.argv[3]
    test_collection = sys.argv[4]
    run_num = sys.argv[5]
    main(train_collection, val_collection, config_path, test_collection, run_num)
    

