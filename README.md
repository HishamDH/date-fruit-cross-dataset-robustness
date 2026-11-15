# Beyond clean accuracy: robustness and cross-dataset generalization of pretrained networks for date fruit variety classification

Code, result files, figures and run logs of the study. Six ImageNet-pretrained networks (ResNet-50, MobileNetV3-L,
EfficientNetV2-S, ConvNeXt-T, DeiT-S, Swin-T) and a classical HOG + LBP + color SVM are trained on one public studio dataset
of nine Saudi date varieties and then tested, without retraining, under simulated corruptions and on phone photographs of
the same varieties from two other public datasets.

## Contents

| Path | What it is |
|---|---|
| `code/` | Data preparation, training, classical baseline, corruption test, external tests, multi-source training, analysis |
| `results/` | One folder per run: `fold<k>.json` (training history, metrics, accuracy under each corruption) and `fold<k>_pred.npz` (test predictions). Also `external_*.json`, `multisource/`, `summary.json` and `tables/` |
| `results_soda/` | The same comparison repeated with SODA (13 varieties) as the training and test set |
| `figures/` | The figures of the paper, written by `code/analyze.py` |
| `logs/` | Console logs of the runs |
| `data_lists/` | The image lists: main dataset (with fold-independent file hashes), the external photographs used, and the SODA whole-fruit selection |

Run names in `results/`: `<model>` is seed 0, `<model>_s1` and `<model>_s2` are the two further seeds, `<model>_rigid` is
the rigid-augmentation recipe, `<model>_strong_s1` is the strong-augmentation recipe, `convnext_noaug*` and
`convnext_scratch` are the ablations. `convnext_in22k_s1`, `vit_in21k_s1`, `dinov2_s1` and `clip_s1` are the networks
pretrained on more data than ImageNet-1k (`*_lr*` are their learning-rate checks). `external_tta_*.json` holds the
test-time remedies (test-time augmentation and batch-statistics adaptation) on the external photographs.

## Datasets (not redistributed here)

1. Main: W. S. N. Alhamdan, "Date Fruit Image Dataset in Controlled Environment", Kaggle,
   `wadhasnalhamdan/date-fruit-image-dataset-in-controlled-environment` (1,658 images, 9 classes).
2. Medina 27: Kaggle, `firasaalsaidi/medina-27-dates-hackathon-dataset-v1` (CC BY 4.0). Six classes are shared with the main dataset.
3. SODA: Kaggle, `gfbati/soda-saudi-arabian-organic-date-fruits-dataset` (CC BY 4.0). Five classes are shared.

## Reproducing

Hardware used: a laptop with an NVIDIA GeForce RTX 2050 (4 GB) and an Intel Core i7-12650H. Software: Python 3.14,
PyTorch 2.14.1 (CUDA 12.6 build), torchvision 0.29.1, timm 1.0.31, scikit-learn 1.8, OpenCV 5.0, rembg (U2-Net).
`code/py.cmd` is a one-line wrapper that runs a script with the project's Python environment; replace it by `python`.

```
python code/prepare_data.py download     fetch the main dataset archive from Kaggle
python code/prepare_data.py build        256 x 256 crops and the manifest
python code/make_masks.py                fruit masks for the background test
python code/run_experiments.py           learning-rate check, 5-fold runs, baseline, ablations, cost, Grad-CAM
python code/followup.py                  rigid-augmentation ablation
python code/replicate_rigid.py           six-model replication with the rigid recipe
python code/run_seeds.py --recipe standard --seeds 1,2
python code/get_external.py kaggle medina27 soda_kaggle
python code/prepare_external.py medina27
python code/prepare_external.py soda
python code/sort_fruit_pit.py            separate whole fruits from pits in SODA
python code/apply_soda_review.py         apply the manual exclusions
python code/prepare_second.py            SODA (13 varieties) as a second dataset
python code/run_queue.py                 second dataset, multi-source training, external tests, ablation seeds
python code/run_queue2.py                third multi-source seed, two further seeds on SODA
python code/run_queue3.py                test-time remedies, larger pretraining, strong augmentation, final external tests
python code/find_repeats.py              repeat photographs of the same fruit in the main dataset
python code/repeats_effect.py
python code/verify_results.py            independent recomputation of the reported numbers
python code/analyze.py                   summary.json, tables and figures
```

Every script skips runs whose result files already exist. Seeds are fixed (folds: 42; fold k of seed s: 1000 + k + 100 s),
but GPU arithmetic is not bit-reproducible, so repeated runs can differ in the last decimal. Checkpoints and the raw
Grad-CAM maps are not stored here because of their size; the scripts recreate them.

## License

The code is released under the MIT License (see `LICENSE`). The datasets keep their own licenses.
