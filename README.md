# CD-TG for Cross-Domain Few-Shot Medical Image Segmentation

Reproduction and evaluation of the **CD-TG (Cross-domain Text-Guided)** plug-in module on cross-domain few-shot medical image segmentation, using three few-shot segmentation backbones (RPMMs, PATNet and DMTNet). Models are trained on natural images (PASCAL VOC) and tested on medical images with only 1 or 5 labelled support examples.

Based on the paper:

> F. Song, Y. Bo, S. Wang, Y. Long, H. Zhang. _Leveraging language to generalize natural images to few-shot medical image segmentation._ Pattern Recognition Letters 201 (2026) 66–72. https://doi.org/10.1016/j.patrec.2026.01.009
> Official code: https://github.com/Lilacis/CD_TG

---

## What this repo contains

| Notebook                                            | What it does                                                                                                                                                                                                                                               |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `notebooks/01_patnet_training_pascal_voc.ipynb`     | Trains PATNet (ResNet-50, fold 0) on PASCAL VOC 2012 + SBD with episodic 1-way 1-shot training. Produces `best_model.pt`, which notebook 02 uses.                                                                                                          |
| `notebooks/02_patnet_cdtg_chest_xray.ipynb`         | Evaluates the trained PATNet on Chest X-Ray (1-shot and 5-shot), then adds the CD-TG module (CLIP ViT-B/16 text encoder, cross-attention, prototype fusion, α = 0.3) and re-evaluates.                                                                     |
| `notebooks/03_dmtnet_cdtg_evaluation.ipynb`         | Evaluates baseline DMTNet on Chest X-Ray, ISIC2018, WBC and CHAOS-MRI, then adds CD-TG and re-evaluates on Chest X-Ray, ISIC2018 and WBC. Custom dataset loaders for Chest X-Ray, WBC and CHAOS-MRI (DICOM) are written inside this notebook.              |
| `notebooks/04_rpmms_cdtg_training_evaluation.ipynb` | A simplified RPMMs-style baseline (ResNet-50 + prototype segmentor) and an RPMMs + CD-TG model (CLIP ViT-B/16 + LoRA + cross-attention + prototype fusion, α = 0.3), both trained on PASCAL VOC and evaluated on ISIC2018, Chest X-Ray, WBC and CHAOS-MRI. |

Outputs are saved in every notebook, so the results can be read without rerunning.

## How CD-TG works (short version)

1. A class description (e.g. "a photo of a nevus ...") is embedded with the CLIP text encoder.
2. The embedding is cross-attended with the support and query features.
3. Mask average pooling on the text-guided support feature gives a text-guided prototype.
4. It is fused with the baseline prototype: `P_final = α·P_text + (1 − α)·P_baseline`.

## Results

All numbers are mIoU (%). "Paper" columns are copied from Table 1 of the CD-TG paper. "Ours" columns come from the notebooks in this repo. Protocols and episode counts differ between notebooks, so compare within a table, not across tables.

### DMTNet (notebook 03)

| Dataset     | Shots | DMTNet (paper) | DMTNet (ours) | + CD-TG (paper) | + CD-TG (ours) |
| ----------- | ----- | -------------- | ------------- | --------------- | -------------- |
| Chest X-Ray | 1     | 73.7           | 63.29         | 77.8            | 60.83          |
| Chest X-Ray | 5     | 77.3           | 64.71         | 79.4            | 61.65          |
| ISIC2018    | 1     | 40.1           | 43.97         | 47.4            | 42.35          |
| ISIC2018    | 5     | 51.1           | 51.96         | 53.7            | 50.81          |
| WBC         | 1     | 48.1           | 79.19         | 50.8            | 79.78          |
| WBC         | 5     | 49.5           | 84.96         | 52.5            | 85.29          |
| CHAOS-MRI   | 1     | 14.3           | 7.98          | 17.3            | not evaluated  |
| CHAOS-MRI   | 5     | 17.2           | 8.16          | 20.2            | not evaluated  |

### PATNet (notebooks 01 and 02, Chest X-Ray only)

| Dataset     | Shots | PATNet (paper) | PATNet (ours) | + CD-TG (paper) | + CD-TG (ours) |
| ----------- | ----- | -------------- | ------------- | --------------- | -------------- |
| Chest X-Ray | 1     | 66.6           | 61.33         | 71.8            | 59.58          |
| Chest X-Ray | 5     | 70.2           | 63.36         | 74.3            | 61.47          |

**PATNet training (notebook 01):** 50 epochs, batch size 12, lr 1e-3, Adam, ResNet-50 backbone, PASCAL VOC fold 0 (11,394 train images, 346 val images). Training episodes were capped at 2,000 per epoch and validation at 200 to fit Kaggle's time limits. The checkpoint with the best validation mIoU is saved.

### RPMMs (notebook 04)

| Dataset     | Shots | RPMMs (paper) | RPMMs (ours) | + CD-TG (paper) | + CD-TG (ours) |
| ----------- | ----- | ------------- | ------------ | --------------- | -------------- |
| ISIC2018    | 1     | 18.0          | 21.15        | 36.2            | 31.74          |
| ISIC2018    | 5     | 20.0          | 20.06        | 37.5            | 33.18          |
| Chest X-Ray | 1     | 30.1          | 30.81        | 40.0            | 49.25          |
| Chest X-Ray | 5     | 30.8          | 26.18        | 40.4            | 41.10          |
| WBC         | 1     | 22.0          | 19.38        | 27.9            | 26.76          |
| WBC         | 5     | 23.6          | 14.50        | 37.2            | 42.41          |
| CHAOS-MRI   | 1     | 10.3          | 0.56         | 15.4            | 0.99           |
| CHAOS-MRI   | 5     | 10.7          | 0.43         | 15.5            | 0.83           |

**Notebook 04 setup:** both models are trained for 10 epochs of about 100 episodes each (AdamW, lr 1e-4, cosine schedule, BCE loss) on PASCAL VOC with SBD masks. The CD-TG model uses 23 pre-written class descriptions in the paper's "A photo of ... in which ..." format. WBC scores average the nucleus and cytoplasm classes, and CHAOS scores average four organs.

## Important notes and limitations

Please read these before comparing with the paper.

- **CD-TG is not trained in notebooks 02 and 03.** The 14 new CD-TG parameters (projection, attention, etc.) are randomly initialised, and only the original PATNet / DMTNet weights are loaded. These CD-TG numbers are inference-only and do not reproduce the paper's gains. In notebook 04, CD-TG is trained.
- **CD-TG did not beat the baseline on Chest X-Ray or ISIC** with PATNet and DMTNet in our runs. It was marginally higher on WBC with DMTNet (+0.3 to +0.6). With the RPMMs model it was higher on most datasets, with the caveats below.
- **Text prompts differ by notebook.** Notebook 03 uses generic prompts of the form `"a photo of organ N"`. Notebook 04 uses 23 pre-written descriptions. Notebook 02 uses the prompts from the patched `CD_TG/encoder.py`. None of them use live GPT-4 output as in the paper.
- **Checkpoint selection in notebook 04 uses ISIC.** The best checkpoint is picked by performance on ISIC, a target-domain dataset, which breaks the strict cross-domain protocol. ISIC results in that notebook are therefore optimistic.
- **The RPMMs baseline is a simplified re-implementation**, not the official RPMMs code, and it is trained for only 10 short epochs. The CD-TG model also adds CLIP features, LoRA and attention layers, so its gain over the baseline cannot be credited to text guidance alone.
- **RPMMs anomalies:** CHAOS scores are under 1% (the pipeline is effectively not working for that dataset), and 5-shot scores fall below 1-shot on Chest X-Ray and WBC for the baseline.
- **WBC numbers in notebook 03 are not comparable to the paper.** We merged the two public WBC subsets into 400 image-mask pairs (the paper reports 200 images) and used our own mask threshold and episode sampling, which likely explains the much higher scores.
- **CHAOS-MRI (notebook 03):** only 247 usable slices were found after filtering (≥ 50 organ pixels), and CD-TG was not run on it.
- **Small evaluations:** notebook 04 evaluates on 200 episodes (ISIC, Chest X-Ray) and 100 per class (WBC, CHAOS), so numbers are noisy. Notebooks 02 and 03 use 600 to 1,000 episodes, which also differs from the original protocol.
- **CLIP weights:** the provided TorchScript CLIP checkpoint was incompatible with `open_clip` in notebook 03, so the official OpenAI ViT-B/16 weights are used there.

## Running the notebooks

All notebooks were run on Kaggle (GPU) and contain Kaggle-specific paths (`/kaggle/input/...`, `/kaggle/working/...`). To rerun elsewhere, update those paths.

**Data**

- PASCAL VOC 2012: https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset (notebook 01) and https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012 (notebook 04)
- SBD augmented masks (`SegmentationClassAug`): https://drive.google.com/uc?id=10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2 (from the PATNet README)
- ISIC2018 Task 1: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation (class split from `CD_TG/dataloader/isic/class_id.csv`)
- Chest X-Ray: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels
- WBC: https://github.com/zxaoyou/segmentation_WBC
- CHAOS-MRI T2-SPIR: https://www.kaggle.com/datasets/anhoangvo/chaos-t1-and-t2 (notebook 04) and https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels (notebook 03)

**Weights and code**

- PATNet code: https://github.com/slei109/PATNet
- DMTNet code and `dmtnet_resnet50.pt`: https://github.com/ChenJiayi68/DMTNet
- CD-TG code: https://github.com/Lilacis/CD_TG
- CLIP ViT-B/16 via `open_clip` (OpenAI weights)

**Main dependencies:** see `requirements.txt` (PyTorch, `open-clip-torch==2.20.0`, `peft`, `loralib`, `gdown`, `SimpleITK`, `pydicom`, `tensorboardX`).

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── 01_patnet_training_pascal_voc.ipynb
│   ├── 02_patnet_cdtg_chest_xray.ipynb
│   ├── 03_dmtnet_cdtg_evaluation.ipynb
│   └── 04_rpmms_cdtg_training_evaluation.ipynb
└── docs/
    ├── DMTNet_Notebook_Explanation.pdf
    └── PATNet_Training_Explanation.docx
```

## Future work

- Train the CD-TG layers for PATNet and DMTNet instead of using random initialisation.
- Use the official RPMMs code, and select checkpoints on a source-domain validation set only.
- Generate class descriptions with GPT-4 as in the paper.
- Make the fusion weight α adaptive instead of fixed at 0.3.
- Run PATNet + CD-TG on ISIC, WBC and CHAOS-MRI.

## Citation

```bibtex
@article{song2026leveraging,
  title   = {Leveraging language to generalize natural images to few-shot medical image segmentation},
  author  = {Song, Feifan and Bo, Yuntian and Wang, Shidong and Long, Yang and Zhang, Haofeng},
  journal = {Pattern Recognition Letters},
  volume  = {201},
  pages   = {66--72},
  year    = {2026},
  doi     = {10.1016/j.patrec.2026.01.009}
}
```

## Acknowledgements

This work builds on RPMMs (Yang et al., ECCV 2020), PATNet (Lei et al., ECCV 2022), DMTNet (Chen et al., IJCAI 2024), CLIP (Radford et al., ICML 2021) and the CD-TG paper above. Please check the licences of the upstream repositories before reusing their code.

## License

MIT License. See the `LICENSE` file.
