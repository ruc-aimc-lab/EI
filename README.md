# Early Intervention (EI)

Code for *EI: Early Intervention for Multimodal Imaging based Disease Recognition* (Findings of CVPR 2026).

## Environment

```bash
pip install -r requirements.txt
```


## Datasets

Download each dataset from its official source and extract under `../imgdata/` (the location the configs expect). The splits used in the paper are already in [datasplit/](datasplit/).

| Dataset | Source |
|---------|--------|
| Derm7pt | <https://derm.cs.sfu.ca/Download.html> |
| MMC-AMD | <https://github.com/li-xirong/mmc-amd> |
| MRNet   | <https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c> |

To use a different location, override `paths.image_root` in the relevant config file.

## Usage

`main.py` runs the full train → predict → evaluate pipeline. General form:

```bash
python main.py <train_collection> <val_collection> <config_path> <test_collection> <gpu>
```

The example below walks through the full Derm7pt workflow with the DINOv2-B backbone. EI is initialized from two single-modal MoR checkpoints, so those have to be trained first.

### Step 1. Single-modal MoR, one config per modality

```bash
# modality 0 (dermoscopic)
python main.py derm7pt_train_balance derm7pt_val \
    configs_uni/derm_dinov2_b_modal0_sgd_mor3_512.json \
    derm7pt_test 0

# modality 1 (clinical)
python main.py derm7pt_train_balance derm7pt_val \
    configs_uni/derm_dinov2_b_modal1_sgd_mor3_512.json \
    derm7pt_test 0
```

Each call writes `out/derm7pt_train_balance/Models/derm7pt_val/<config_name>/runs_<n>/best_model.pkl`. The run index `n` auto-increments on each invocation, so running each single-modal config three times gives `runs_0`, `runs_1`, `runs_2`.

The same loop is wrapped in [derm_main.sh](derm_main.sh):

```bash
bash derm_main.sh configs_uni/derm_dinov2_b_modal0_sgd_mor3_512.json 0
bash derm_main.sh configs_uni/derm_dinov2_b_modal1_sgd_mor3_512.json 0
```

### Step 2. EI initialized from the single-modal checkpoints above

```bash
python main.py derm7pt_train_balance derm7pt_val \
    configs_mul/derm_dinov2_b_ei_modal01_all_sgd_ft_512.json \
    derm7pt_test 0
```

Or the 3-seed wrapper:

```bash
bash derm_main.sh configs_mul/derm_dinov2_b_ei_modal01_all_sgd_ft_512.json 0
```

The EI config picks up the single-modal checkpoints via its `model_params.pretrained_configs` / `pretrained_ckpts` fields.

The MMC-AMD and MRNet workflows are identical. Swap the dataset prefix in the config filenames and use `amd_main.sh` / `mrnet_main.sh`. Use `openclip_vit_b` instead of `dinov2_b` in the config name to switch backbones. The MRNet EI config uses `modal012` since all three views (sagittal, axial, coronal) are trained jointly.

Multi-GPU is supported via `accelerate`: pass `"0,1"` for the gpu argument in the `*_main.sh` runners.



## Citation

```
@inproceedings{wei2026ei,
  title     = {{EI}: Early Intervention for Multimodal Imaging based Disease Recognition},
  author    = {Wei, Qijie and Lin, Hailan and Li, Xirong},
  booktitle = {Findings of CVPR},
  year      = {2026}
}
```