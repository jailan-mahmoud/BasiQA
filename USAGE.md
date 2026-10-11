# BasiQA / FTHNet – Usage Guide (Test & Train)

Exact, step-by-step commands for running inference (testing) and training FTHNet.
All commands are run from the **repository root** (`BasiQA-main/`).

> Commands are given for **PowerShell (Windows)** first, with the Linux/macOS (bash) equivalent where it differs.
> The only difference is how the GPU is selected.

| Task | Shell | Select GPU 0 + run |
|------|-------|--------------------|
| any  | PowerShell | `$env:CUDA_VISIBLE_DEVICES="0"; python ...` |
| any  | bash       | `CUDA_VISIBLE_DEVICES=0 python ...` |

For CPU-only runs, set `num_gpu: 0` in the YAML you use (see [Troubleshooting](#troubleshooting)).

---

## 0. One-time setup

### 0.1 Create the environment

```powershell
conda create -n basiqa python=3.8 -y
conda activate basiqa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Use a different `cuXXX` suffix if your CUDA version differs.

### 0.2 Install SoftPool (required by FTHNet)

Follow <https://github.com/alexandrosstergiou/SoftPool>.

### 0.3 Check the pretrained weights are in place

Required file: `pretrained_weight/net_g_226264S4.pth`

```powershell
Test-Path .\pretrained_weight\net_g_226264S4.pth     # must print True
```

If `False`, download it (Google Drive: <https://drive.google.com/drive/folders/1gXaa77aARo1sdqky3_81JD6ofL17fGUU?usp=sharing>, or Baidu code `fth1`) and put it in `pretrained_weight/`.

### 0.4 Check the dataset layout

```
datasets/
├── sample_dataset/                  # tiny set (5 images) for pipeline checks
│   ├── images/
│   └── cross_validation-00-test.xlsx
└── FIQSDataset/                     # full FIQS dataset (2246 images)
    ├── images/                      # 384-px images used by the configs
    ├── full_size_images/
    ├── meta_data.xlsx
    ├── dataset_fetch.py             # regenerates the 10-fold splits
    └── divisions/
        └── cross_validation-{00..09}-{train,val,test}.xlsx
```

Split files must have the columns `file_name` and `meanOpinionScore` (score 0–100; `full_score: 100` in the YAML normalises it).

---

## 1. Testing (inference + metrics)

Config: [options/test/test_FTHNet.yml](options/test/test_FTHNet.yml)
Entry point: [basiqa/test.py](basiqa/test.py)

### 1.1 Quick test on the sample dataset (works as shipped)

```powershell
$env:CUDA_VISIBLE_DEVICES="0"; python ./basiqa/test.py -opt ./options/test/test_FTHNet.yml
```
```bash
CUDA_VISIBLE_DEVICES=0 python ./basiqa/test.py -opt ./options/test/test_FTHNet.yml
```

Reference result from a previous run on the sample set: SRCC 0.9436, PLCC 0.9750, L1 3.68, RMSE 0.0449.

### 1.2 Test on the full FIQS dataset (one fold)

Edit the `datasets.test_1` block in [options/test/test_FTHNet.yml](options/test/test_FTHNet.yml):

```yaml
datasets:
  test_1:
    name: FIQS
    type: ScoreImageDataset3
    image_folder: ./datasets/FIQSDataset/images
    csv_path: ./datasets/FIQSDataset/divisions/cross_validation-00-test.xlsx
```

Then run the same command as 1.1.

### 1.3 Test your own trained checkpoint

In the same YAML change:

```yaml
path:
  pretrain_network_g: experiments/<exp_name>/models/0_net_g_latest.pth
```

Then run the command from 1.1. (Use the same `network_g` block that the model was trained with.)

### 1.4 Test on a custom dataset

1. Put images in `datasets/<mydata>/images/`.
2. Create `datasets/<mydata>/score.xlsx` (or `.csv`) with columns `file_name`, `meanOpinionScore`.
3. Point `image_folder` and `csv_path` in the test YAML to them.
4. Run the command from 1.1.

You can also use the other ready-made test configs:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"; python ./basiqa/test.py -opt ./options/test/IQA/test_hyper.yml
```

### 1.5 Where the output goes

```
results/<name>/
├── test_<name>_<timestamp>.log                  # full log + final metrics
└── visualization/<dataset_name>/
    ├── prediction__.csv                         # per-image prediction vs ground truth (+ metrics)
    └── results.csv                              # predictions averaged over augment_ratio runs (0–100 scale)
```

Metrics printed at the end: `L1`, `SRCC`, `PLCC`, `RMSE`, `PRsum`.

### 1.6 Quick test vs. test on the full dataset

Both use the exact same command and the same pretrained model. Only the two dataset lines in the YAML differ.

| | Quick test (1.1) | Full-dataset test (1.2) |
|---|---|---|
| `image_folder` | `./datasets/sample_dataset/images` | `./datasets/FIQSDataset/images` |
| `csv_path` | `sample_dataset/cross_validation-00-test.xlsx` | `FIQSDataset/divisions/cross_validation-00-test.xlsx` |
| Images evaluated | 5 | The whole test split of fold 00 (hundreds of images; see the "Number of test images" line in the log, which is the split size × `augment_ratio`) |
| Purpose | Check that the environment, weights and pipeline run end to end | Get a meaningful quality estimate |
| Trustworthy metrics? | **No.** SRCC/PLCC over 5 images are extremely noisy and say nothing about model quality | Yes, for that one fold |
| Runtime | Seconds (about 15 s in the earlier run) | Minutes, since each image is evaluated `augment_ratio` (20) times |

Use the quick test only as a smoke test. Never report its numbers.

> Note on `augment_ratio: 20`: every image is listed 20 times, and each pass uses a **random crop** (and flip). `results.csv` averages the 20 predictions per image. Because the crops are random, the metrics change slightly (typically in the 3rd decimal) between runs unless you fix the seed (see [3.3](#33-make-runs-repeatable)).

### 1.7 One fold vs. cross-validation testing

`datasets/FIQSDataset/divisions/` holds **10 different train/val/test splits** of the same 2246 images (`cross_validation-00-…` to `-09-…`). Each fold has its own held-out **test** file.

| | One fold | Cross-validation (10 folds) |
|---|---|---|
| What it is | A single split, e.g. fold `00` | All 10 splits, each with its own model |
| Test images covered | About 20% of the dataset (one test file) | Every fold's test set; across folds, the images are covered many times |
| Models needed | 1 | 10 (one trained per fold, each evaluated on its own fold's test file) |
| Result | One number per metric. Depends on which images landed in that split | Mean ± std over 10 folds. Much more reliable and is what a paper should report |
| Cost | 1 test run | 10 training runs + 10 test runs |
| Commands | [1.2](#12-test-on-the-full-fiqs-dataset-one-fold) | [3.2](#32-cross-validation-test-10-folds) |

In short: a one-fold result tells you how the model did on one particular split. Cross-validation averages over 10 splits so a lucky or unlucky split does not decide the result.

> Important: the same fold index must be used for training and testing. Training on fold `03` and testing on fold `07`'s test file can leak: the model may have seen those test images during training, which inflates the scores.

---

## 2. Training

Entry point: [basiqa/train_multi.py](basiqa/train_multi.py)
Experiment folders are created automatically under `experiments/<name>/` and TensorBoard logs under `tb_logger/<name>/`.

### 2.1 Smoke test (sample dataset, 2000 iterations, batch 4)

Uses [options/train/IQA/train_fthnet_trial.yml](options/train/IQA/train_fthnet_trial.yml). It trains and validates on the sample set and fine-tunes from the pretrained weights.

```powershell
$env:CUDA_VISIBLE_DEVICES="0"; python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_trial.yml
```
```bash
CUDA_VISIBLE_DEVICES=0 python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_trial.yml
```

If `experiments/fth_trial_ex1/` already exists, it is renamed to `fth_trial_ex1_archived_<timestamp>` and a fresh run starts.

### 2.2 Train / fine-tune on the full FIQS dataset

The shipped config [options/train/IQA/train_fthnet_std_multi.yml](options/train/IQA/train_fthnet_std_multi.yml) points to `./datasets/data/score_list_*.csv`, which **does not exist in this repo**. Edit the following before running.

**a) Dataset paths** (in `datasets.train` and `datasets.val`):

```yaml
datasets:
  train:
    image_folder: ./datasets/FIQSDataset/images
    csv_path: ./datasets/FIQSDataset/divisions/cross_validation-00-train.xlsx
  val:
    image_folder: ./datasets/FIQSDataset/images
    csv_path: ./datasets/FIQSDataset/divisions/cross_validation-00-val.xlsx
```

**b) Network type**: the std config uses `type: FTHNet3`, but the provided pretrained weights were used with `FTHNet4` (see the trial and test configs). Set:

```yaml
network_g:
  type: FTHNet4
```

**c) Pretrained weights** (add/confirm under `path`):

```yaml
path:
  pretrain_network_g: pretrained_weight/net_g_226264S4.pth
  strict_load_g: false
  resume_state: ~
```
Remove `pretrain_network_g` (set to `~`) to train from scratch.

**d) Rounds**: `multitrain: true` and `round: 10` repeat the full training 10 times with `dataset_random_fetch: false` (the same CSVs are reused each round). For a single run set:

```yaml
multitrain: false     # or: round: 1
```

**e) Batch size / workers**: `batch_size_per_gpu: 24` and `num_worker_per_gpu: 4` need a large GPU. Lower them (e.g. `8` and `0`) if you run out of memory. On Windows `num_worker_per_gpu: 0` is the safest.

**Run:**

```powershell
$env:CUDA_VISIBLE_DEVICES="0"; python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_std_multi.yml
```
```bash
CUDA_VISIBLE_DEVICES=0 python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_std_multi.yml
```

### 2.3 Train on a custom dataset

```powershell
Copy-Item .\options\train\IQA\train_fthnet_std_multi.yml .\options\train\IQA\my_config.yml
# edit my_config.yml: name, image_folder, csv_path (train + val), pretrain_network_g
$env:CUDA_VISIBLE_DEVICES="0"; python ./basiqa/train_multi.py -opt ./options/train/IQA/my_config.yml
```

Change `name:` in the YAML so the experiment does not collide with an existing one.

### 2.4 Override YAML values from the command line (no file edit)

```powershell
python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_trial.yml --force_yml train:total_iter=500 datasets:train:batch_size_per_gpu=2 name=my_quick_run
```

### 2.5 Resume an interrupted run

Automatic (picks the newest `.state` in `experiments/<name>/training_states/`):

```powershell
python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_trial.yml --auto_resume
```

Manual: set `path.resume_state` in the YAML:

```yaml
path:
  resume_state: experiments/fth_trial_ex1/training_states/500.state
```

### 2.6 Debug mode (fast, no TensorBoard/archiving)

Use a `name` containing `debug` or pass the flag:

```powershell
python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_trial.yml --debug
```

### 2.7 Other available training configs

All under [options/train/IQA/](options/train/IQA/), run with the same command pattern:

| Folder | Purpose |
|--------|---------|
| `bb/` | Backbone comparison (ConvNeXt, MSG, ViT) |
| `bs/` | Batch size ablation |
| `depth/` | Swin depth / embed-dim ablation |
| `loss/` | L1 / L2 / mixed loss ablation |
| `lr/` | Learning-rate and schedule ablation |
| `normlize/` | Normalisation / crop ablation |
| `hypernet/` | Hypernetwork vs. non-hypernetwork, classification variants |
| `tail/`, `whole/`, `distortionpn/` | Other ablations |

Classification-variant configs (`hypernet/*class*.yml`) use [basiqa/train_class.py](basiqa/train_class.py) instead:

```powershell
python ./basiqa/train_class.py -opt ./options/train/IQA/hypernet/train_hyper_swin_class.yml
```

### 2.8 Monitor training

```powershell
tensorboard --logdir ./tb_logger
```
Open <http://localhost:6006>.

### 2.9 Where the output goes

```
experiments/<name>/
├── train_<name>_<timestamp>.log          # log + validation metrics
├── train_<config>.yml                    # copy of the config used
├── models/
│   ├── 0_net_g_<iter>.pth                # periodic checkpoints (every save_checkpoint_freq)
│   ├── 0_net_g_latest.pth
│   └── (best checkpoint when save_best: true)
└── training_states/<iter>.state          # for resuming
tb_logger/<name>/                         # TensorBoard events
```

Use `experiments/<name>/models/0_net_g_latest.pth` (or the best checkpoint) for testing, see [1.3](#13-test-your-own-trained-checkpoint).

### 2.10 (Optional) Regenerate the 10-fold splits

```powershell
cd datasets\FIQSDataset
python dataset_fetch.py
cd ..\..
```
This overwrites `divisions/cross_validation-*.xlsx` with new random splits (70% train / 10% val / 20% test).

---

## 3. Reproducing the paper's results

The repository does not contain the paper's numbers (the paper is still under revision), so there is nothing here to compare against automatically. What follows is the protocol that matches how the code is built. Compare its output with the tables in the paper yourself.

### 3.1 What decides whether your numbers match

| Factor | What to do |
|--------|-----------|
| Weights | Use `pretrained_weight/net_g_226264S4.pth` (the authors' released checkpoint) or train your own. |
| Network | `network_g` must be exactly `FTHNet4`, `embed_dim: 64`, `depths: [2,2,6,2]`, `num_heads: [2,4,8,16]`, `window_size: 12`. |
| Data | The full FIQS dataset (2246 images) and its `divisions/` splits, **not** `sample_dataset`. |
| Preprocessing | Keep the test YAML values: `image_size: 384`, `resize: true`, `square_cut: true`, `full_score: 100`, `augment_ratio: 20`, `mean`/`std` as given. |
| Metrics | Take `SRCC`, `PLCC` (and `L1`, `RMSE`) from the end of the test log. |
| Randomness | Test crops are random, so fix the seed (3.3). Residual differences in the 3rd decimal are expected. |

### 3.2 Cross-validation test (10 folds)

Protocol: for each fold `k` in 00..09, train a model on `cross_validation-0k-train.xlsx`, validate on `-val.xlsx`, test on `-test.xlsx`, then report the mean ± std of the 10 test results.

**Step 1 – train one model per fold (PowerShell).** Use a single-run config (`multitrain: false`, edited as in [2.2](#22-train--fine-tune-on-the-full-fiqs-dataset)) and override the fold and name from the command line:

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
foreach ($k in 0..9) {
  $f = "{0:D2}" -f $k
  python ./basiqa/train_multi.py -opt ./options/train/IQA/train_fthnet_std_multi.yml --force_yml name=fth_cv$f datasets:train:csv_path=./datasets/FIQSDataset/divisions/cross_validation-$f-train.xlsx datasets:val:csv_path=./datasets/FIQSDataset/divisions/cross_validation-$f-val.xlsx
}
```

**Step 2 – test each model on its own fold's test file:**

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
foreach ($k in 0..9) {
  $f = "{0:D2}" -f $k
  python ./basiqa/test.py -opt ./options/test/test_FTHNet.yml --force_yml name=FTHNet_cv$f datasets:test_1:image_folder=./datasets/FIQSDataset/images datasets:test_1:csv_path=./datasets/FIQSDataset/divisions/cross_validation-$f-test.xlsx path:pretrain_network_g=experiments/fth_cv$f/models/0_net_g_latest.pth
}
```

Each run writes `results/FTHNet_cv<k>/test_FTHNet_cv<k>_<timestamp>.log`.

**Step 3 – collect the results.** Read the last block of each log (`SRCC`, `PLCC`, ...) and average over the 10 folds:

```powershell
Select-String -Path .\results\FTHNet_cv*\test_*.log -Pattern "# (SRCC|PLCC|L1|RMSE)"
```

### 3.3 Make runs repeatable

Uncomment the seed line in the YAML you use (test and train):

```yaml
manual_seed: 114514
```

Otherwise a new random seed is drawn on each run and the random crops differ.

### 3.4 Using the released checkpoint only (no training)

Test the checkpoint on a fold's test file as in [1.2](#12-test-on-the-full-fiqs-dataset-one-fold) (or loop over all 10 test files with the Step 2 command, keeping `pretrain_network_g: pretrained_weight/net_g_226264S4.pth`).

Caveat: it is not documented which images the released checkpoint was trained on. If it was trained on images that appear in a given fold's test file, those scores will be optimistically high. For a clean cross-validation, retrain per fold as in 3.2.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate the env (`conda activate basiqa`) and re-run `pip install -r requirements.txt`. |
| `No module named softpool` / SoftPool error | Complete step 0.2. |
| `FileNotFoundError` for CSV/images | Fix `image_folder` / `csv_path` in the YAML; paths are relative to the repo root. |
| CUDA out of memory | Lower `batch_size_per_gpu`, or `image_size` only if you accept different results. |
| No GPU available | Set `num_gpu: 0` in the YAML and omit `CUDA_VISIBLE_DEVICES`. Note: resuming a `.state` file in `train_multi.py` assumes CUDA. |
| Dataloader hangs / errors on Windows | Set `num_worker_per_gpu: 0`. |
| Size mismatch when loading weights | `network_g` in the YAML must match the checkpoint (`FTHNet4`, `embed_dim: 64`, `depths: [2,2,6,2]`, `window_size: 12`); `strict_load_g: false` tolerates extra/missing keys. |
