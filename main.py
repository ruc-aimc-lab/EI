from train import main as train_main
from predict import main as predict_main
from evaluate import main as evaluate_main
import sys
import os

# import tempfile
# tempfile.tempdir = _tmpdir

def main(train_collection, val_collection, config_path, test_collection, device):
    run_num = train_main(train_collection, val_collection, config_path, device)

    if device == '-1':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_rank != 0:
            return

    predict_main(train_collection, val_collection, config_path, test_collection, run_num, device)
    evaluate_main(train_collection, val_collection, config_path, test_collection, run_num)
    

if __name__ == '__main__':
    train_collection = sys.argv[1]
    val_collection = sys.argv[2]
    config_path = sys.argv[3]
    test_collection = sys.argv[4]
    device = sys.argv[5]
    main(train_collection, val_collection, config_path, test_collection, device)
    