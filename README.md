# An Instrument for Small Polyp Localization in Colonoscopy Images Using a Task-Aware Progressive Fusion Network (TAPFuseNet) 🚀

Official implementation of **TAPFuseNet**, a **small polyp segmentation model** for colonoscopy images.

## 🔗 Quick Links

- 📦 **Model Weights (Google Drive)**: [Download](https://drive.google.com/file/d/1samxCNWUpSm5jrjoSPY6RQ26VDO9y5sd/view?usp=sharing)
- 🧩 **SAM-ViT Pretrained Checkpoint (Google Drive)**: [Download](https://drive.google.com/file/d/1GEiteVcAO5PWR7LJTG16jkEPq0LWnZ90/view?usp=drive_link)
- 🖼️ **Predicted Masks (Google Drive)**: [Download](https://drive.google.com/file/d/1ga8mIZ1k5p33u1GbQxIbNjtu2wqKMNvy/view?usp=drive_link)
- 🗂️ **Dataset (Google Drive)**: [Download](https://drive.google.com/file/d/1sO4AV3EmGZkK5jjXEYLOoTDjlgk_nblT/view?usp=drive_link)

## 🧠 Project Structure

- `model/`: model components (`tapfusenet.py`, encoder, decoder, transformer, blocks)
- `dataset/`: dataset loader
- `utils/`: loss and utility meters
- `train.py`: training script
- `evaluation.py`: evaluation script
- `requirements.txt`: package dependencies

## 🛠️ Environment Setup

```bash
pip install -r requirements.txt
```

## 📚 Dataset Preparation (PraNet Setup)

This project follows the **PraNet-style dataset organization**.

```text
dataset/
  SOD/
    Kvasir/
      train/
        image/
        mask/
      test/
        image/
        mask/
    CVC-ClinicDB/
      test/
        image/
        mask/
    CVC-ColonDB/
      test/
        image/
        mask/
    ETIS-Lari/
      test/
        image/
        mask/
```

If you download dataset files from Drive, extract and place them exactly as above.

## 🏋️ Training

```bash
python train.py
```

Before training, check/update paths in `train.py`:
- `Args.data_path`
- `Args.encoder_ckpt`
- `Args.save_dir`

Backbone initialization weights (for training):
- 🧩 [Download](https://drive.google.com/file/d/1GEiteVcAO5PWR7LJTG16jkEPq0LWnZ90/view?usp=drive_link)

## 🔍 Evaluation / Inference

```bash
python evaluation.py
```

Before evaluation, check/update paths in `evaluation.py`:
- `Args.checkpoint`
- `Args.data_path`
- `Args.result_path`

Outputs:
- predicted masks: `results/<dataset>/`
- metrics CSV: `results/sod_metrics.csv`

## 📊 Precomputed Predictions

If you only need results without running inference:
- 🖼️ [Download](https://drive.google.com/file/d/1ga8mIZ1k5p33u1GbQxIbNjtu2wqKMNvy/view?usp=drive_link)

## 📥 Pretrained Weights

To run inference directly with my trained model:
1. Download checkpoint: [Model Weights](https://drive.google.com/file/d/1samxCNWUpSm5jrjoSPY6RQ26VDO9y5sd/view?usp=sharing)
2. Put the `.pth` file in `output/` (or any path you prefer)
3. Set `Args.checkpoint` in `evaluation.py`
4. Run `python evaluation.py`

## ✅ Notes

- Third-party attributions: see `THIRD_PARTY_NOTICES.md`.
- Keep large files (datasets, checkpoints, results) out of git; `.gitignore` is included.
- Add your repository `LICENSE` file before public release.
