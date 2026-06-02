import os
import sys
import json
from trainer import Trainer
import warnings
warnings.filterwarnings("ignore")



def main(train_collection, val_collection, config_path, device):
    f = open(config_path, 'r').read()
    config = json.loads(f)

    paths = config['paths']
    training_params = config['training_params']
    augmentation_params = config['augmentation_params']
    paths['train_collection'] = train_collection
    paths['val_collection'] = val_collection
    paths['config_path'] = config_path
    
    trainer = Trainer(paths=paths,
                      training_params=training_params,
                      augmentation_params=augmentation_params,
                      device=device
                      )
    trainer.train()
    return trainer.run_num


if __name__ == '__main__':
    train_collection = sys.argv[1]
    val_collection = sys.argv[2]
    config_path = sys.argv[3]
    device = sys.argv[4]

    main(train_collection, val_collection, config_path, device)

