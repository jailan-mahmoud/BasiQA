# Options (YAML configs)

Every run is driven by one YAML file passed with `-opt`. The file sets the dataset, model architecture, loss, optimizer, schedule, validation metrics and logging.

```
options/
├── test/                  # evaluation configs  -> basiqa/test.py
│   ├── test_FTHNet.yml    # FTHNet (FTHNet4) on the sample dataset
│   └── IQA/               # HyperSwin evaluations (internal / EyeQ / DeepDRiD / DRIMDB)
└── train/IQA/             # training configs
    ├── train_fthnet_std_multi.yml   # standard full FTHNet run
    ├── train_fthnet_multi_dummy.yml # smoke test of the multi-round pipeline
    └── bb/ bs/ depth/ loss/ lr/ normlize/ distortionpn/ hypernet/ tail/ whole/   # ablations
```

## How to run

```bash
# Train (regression / MOS)
python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_std_multi.yml
# Train (3-class classification configs, e.g. hypernet/*_class*.yml)
python ./basiqa/train_class.py -opt ./options/train/IQA/hypernet/train_hyper_swin_class.yml
# Test
python ./basiqa/test.py -opt ./options/test/test_FTHNet.yml
```

Select the GPU with `CUDA_VISIBLE_DEVICES` (see [USAGE.md](../USAGE.md)).

## Naming conventions

| Word in filename | Meaning |
|---|---|
| `multi` / `muliti` | Multi-round training: the `round` key sets how many times the model is trained, each round with a fresh train/val split. Run with `train_multi.py`. |
| `dummy` | Tiny smoke-test config (about 100 iterations, small split). It only checks that the pipeline runs and gives no real results. |
| `std` | Standard full-length run. |
| `class` | 3-class quality classification (cross-entropy, `EyeQSClassDataset`) instead of score regression. |

## Ablation folders (`train/IQA/`)

| Folder | What varies |
|---|---|
| `bb` | Backbone (ConvNeXt, MSG, ViT) |
| `bs` | Batch size |
| `depth` | Swin depths and embed width (file name `depths_width`, e.g. `2262_64`) |
| `loss` | L1, L2, SmoothL1 and mixed losses |
| `lr` | Learning rate and schedule (`lrsoft` = softer schedule with warmup) |
| `normlize` | Input normalization and cropping |
| `distortionpn` | Pooling variants (avg, soft, dpw) |
| `hypernet` | Hyper-network vs. non-hyper-network baselines (including `*_class*` classification configs) |
| `tail`, `whole` | Which parts of the network are tuned / model size variants |

## Key fields

- `name`: experiment name. Outputs go to `experiments/<name>/`, test results to `results/<name>/`.
- `model_type`: `HyperIQAModel` (regression), `HyperClassModel` / `FClassModel` (classification).
- `round`, `val_ratio`, `dataset_random_fetch`, `random_fetch_path`: multi-round splitting. With `dataset_random_fetch: true`, each round re-splits the CSV at `random_fetch_path`.
- `datasets.*`: `type`, `image_folder`, `csv_path`, batch size, augmentation.
- `network_g`: architecture (`FTHNet3`, `FTHNet4`, `HyperSwin4`, ...). `path.pretrain_network_g` loads weights.
- `train`: optimizer, LR milestones, `total_iter`, loss.
- `val`: `val_freq` and metrics (L1, SRCC, PLCC, Accuracy, ...).
- `logger`: `print_freq`, `save_checkpoint_freq`.

## Dataset paths

All train configs and `test/IQA/test_hyper.yml` point to the FIQS dataset:

- `image_folder: ./datasets/FIQSDataset/images`
- Configs with `dataset_random_fetch: true` re-split `random_fetch_path: ./datasets/FIQSDataset/meta_data.xlsx` (all 2246 images) every round. `val_ratio` sets the validation share, and each split is saved under `experiments/<name>/models/`.
- `csv_path` points to `divisions/cross_validation-00-{train,val}.xlsx`. It is only used when `dataset_random_fetch: false`; otherwise it is overwritten by the random split.
- The external test configs use `./datasets/eyeq`, `./datasets/deepdrid` and `./datasets/drimdb`. Those datasets are not included in this repo.

## Before you run a config

- `train_fthnet_std_multi.yml` has `dataset_random_fetch: false`, so it reads the fold files in `divisions/`. Those files list `.jpeg` names, but `images/` contains `.PNG` files. Fix the extension before using them, or set `dataset_random_fetch: true`.
- `network_g.type` in `train_fthnet_std_multi.yml` is `FTHNet3`, but the released weights were trained with `FTHNet4`.
