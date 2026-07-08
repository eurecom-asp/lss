# Latent Secret Spin (LSS)

### Keyed Orthogonal Rotations for Blind Speech Watermarking in Anisotropic Latent Spaces

Latent Secret Spin (LSS) is a latent-space speech watermarking framework designed for neural audio codecs. Instead of embedding information directly in the waveform, LSS modifies the codec's latent representations, enabling robust, minimally perceptible watermarking while remaining compatible with modern neural speech generation and compression pipelines.

<p align="center">
  <img src="docs/lss_pipeline.jpeg" width="900"/>
</p>

The framework provides a complete pipeline for PCA estimation, watermark embedding, decoding, re-encoding, and watermark detection. It includes utilities for PCA-based latent modeling, secret-key generation, payload embedding, and both latent-domain and waveform-domain detection. LSS supports reproducible research and adapts to different neural codec architectures with minimal changes.

## 💡 Theory Principles and Implementation

Traditional watermarking systems modify the raw speech waveform or frequency spectrum. In contrast, LSS operates on the geometric structure of the latent representation rather than the waveform itself. This design improves robustness to common signal degradations while minimizing perceptual quality degradation. The framework also requires no neural network (re)training.

The algorithm relies on four core stages:

### 1. Latent Extraction

The LSS framework passes raw speech through a pre-trained neural encoder (EnCodec). The encoder compresses the speech into continuous latent feature maps.

### 2. PCA Projection

The system projects these latent features into a Principal Component Analysis (PCA) space. This coordinate system decorrelates the features and isolates the variance. The resulting space is anisotropic: different principal components contain different amounts of signal energy. LSS exploits this anisotropy to identify suitable embedding directions for watermark insertion within the PCA domain.

### 3. Keyed Rotation (Embedding)

The embedding process uses a secret cryptographic key to generate a pseudo-random schedule. This schedule selects specific pairs of principal components (geometric planes) across different time chunks.

The system applies small orthogonal rotations (_Givens rotations_) to these planes to encode the binary payload. Rotating two principal axes with different variances introduces a measurable cross-covariance between them. This creates a latent signature that can later be recovered during detection.

For small rotation angles, the induced covariance is approximately proportional to the rotation angle. Consequently, its sign reveals the direction of the applied rotation and serves as the fundamental watermark carrier.

### 4. Blind Detection

The detector does not require access to the original speech recording. Instead, it extracts latent features from a suspect audio file and reconstructs the embedding schedule using the secret key.

The system isolates the targeted geometric planes and computes their cross-covariance values. If the correct key is supplied, the extracted covariance terms sum constructively and produce a strong detection score. If the key is incorrect, or if the audio does not contain a watermark, the covariance contributions tend to cancel out and yield a score close to zero.


## 🗂️ Repository Structure

This repository contains the **official reference implementation** of **Latent Secret Spin (LSS)** together with the evaluation protocol described in our paper, *Latent Secret Spin: Keyed Orthogonal Rotations for Blind Speech Watermarking in Anisotropic Latent Spaces*, published in **Odyssey 2026**. Please see the [Citation](#-citation) section if you use this repository in your research.

* `config/config_LSS.yaml` — Central configuration file controlling paths, payload bits, secret keys, and model parameters.
* `src/` — Core implementation modules (`audio_processing.py`, `encodec_wrapper.py`, `lss_pca.py`, `utils.py`).
* `estimate_pca.py` — Estimates the PCA basis from a dataset of clean speech.
* `LSS_embedding.py` — Embeds the secret payload into selected speech audio samples.
* `apply_manipulations.py` — Applies signal distortions to evaluate watermark robustness (optional).
* `LSS_detection.py` —  Aggregates detection results, computes evaluation metrics, and generates detection summary statistics.

## ⚙️ Setup and Configuration

Configure your base settings in `config/config_LSS.yaml` before running the tools.

You must define:

* `paths.pca_estim_audio` — Target folder containing PCA training audio.
* `run_lss.key` — Secret cryptographic hexadecimal key (minimum 16 bytes).
* `script.bits` — Binary payload sequence to embed (e.g., `"10011101"`).

The tools operate as standalone scripts. Configuration values can also be overridden via command-line arguments.

## 🚀 Quick Start

Clone the repository and install the required dependencies.

```bash
git clone ...
cd LSS
pip install -r requirements.txt

python estimate_pca.py
python LSS_embedding.py
python LSS_detection.py
```

### 1. 📊 Estimate the PCA Basis

Before watermark embedding, LSS must estimate the principal components of the target speech distribution. This PCA basis defines the geometric space used for latent rotations.

```bash
python estimate_pca.py -i /path/to/raw_dataset/ -o ./checkpoints/ --max_files 25000
```

### 2. 🔐 Embed the LSS Watermark

Run the embedding script to inject a binary payload into a collection of speech files.

```bash
python LSS_embedding.py -i /path/to/input_speech/ -o ./path/to/watermarked_output/
```

You can override the secret key, payload bits, temporal framing, and other configurable parameters directly from the command line.

```bash
python LSS_embedding.py \
    -i ./path/to/input_speech/ \
    -o ./path/to/watermarked_output/ \
    --key "0123456789abcdef" \
    --bits "11110000"
```

### 3. 🛠️ Apply DSP Manipulations

To evaluate watermark robustness, LSS provides a utility for simulating common signal processing distortions. The script organizes manipulated outputs into subdirectories according to the selected attack type.

Supported attacks include:

`lowpass_hz`, `highpass_hz`, `bandpass_hz`, `mp3_kbps`, `resampling`, `white_noise`, and `pink_noise`.

```bash
python apply_manipulations.py \
    -i ./watermarked_speech/ \
    -o ./attacked_speech/ \
    --attack mp3_kbps \
    --param 128 \
    --save_manipulated_wav
```

### 4. 🔎 Detect the Watermark

The detection script supports two operating modes.

#### Benchmarking Mode

Provide both the suspect speech directory and the corresponding clean reference directory. The script computes the Equal Error Rate (EER), Perceptual Evaluation of Speech Quality (PESQ), and additional evaluation statistics.

It also estimates an operating threshold based on the EER, stores reports in the results directory, and prints a summary classification table.

```bash
python LSS_detection.py \
    -i ./path/to/suspect_watermarked_speech/ \
    -r ./path/to/corresponding_clean_speech/
```

#### Binary Classification Mode

Provide a directory of suspect speech files together with a known decision threshold. No clean reference directory is required.

The script extracts latent signatures and produces binary decisions (`WATERMARKED` or `CLEAN`) for each audio sample.

```bash
python LSS_detection.py \
    -i ./path/to/suspect_watermarked_speech/ \
    --threshold 2.45
```

#### On-the-Fly Manipulation

Signal distortions can also be applied during detection without saving manipulated audio to disk.

```bash
python LSS_detection.py \
    -i ./path/to/suspect_watermarked_speech/ \
    -r ./path/to/corresponding_clean_speech/ \
    --attack white_noise \
    --manip_param 10
```

## 📈 Detection Threshold Calibration

The detection script outputs two values for each speech sample:

1. A continuous watermark presence score.
2. A binary classification (`WATERMARKED` or `CLEAN`).

The binary decision depends on the selected `--threshold` value.

> **Important:** Do not set the threshold to zero. Clean, unwatermarked speech naturally produces a distribution of scores centered near zero, including small positive values. A threshold of zero will therefore generate excessive false positives.

To determine a suitable threshold, run the detector in **Benchmarking Mode** on a representative dataset containing both clean and watermarked speech. The script computes the Equal Error Rate (EER), which identifies the operating point where the false acceptance rate and false rejection rate are equal.

This EER-derived threshold provides a practical baseline for deployment in **Binary Classification Mode**.

Threshold selection ultimately determines the detector's operating behavior:

* **Increase the threshold** to prioritize security and reduce false positives. This may, however, reduce sensitivity to heavily degraded watermarks.
* **Decrease the threshold** to prioritize sensitivity and improve recovery of degraded watermarks. This increases the risk of incorrectly classifying clean audio as watermarked.

## Installation and Dependencies

Install the core Python packages using the requirements file:

```bash
pip install -r requirements.txt
```

### System Prerequisites

* **PyTorch (CUDA):** A standard `pip` install often downloads the CPU version of PyTorch. To use GPU acceleration, you must install PyTorch with the correct CUDA drivers directly from the [PyTorch website](https://pytorch.org/).
* **FFmpeg:** The audio processing script uses FFmpeg for MP3 attacks. You must install FFmpeg on your operating system (e.g., `apt install ffmpeg` or `brew install ffmpeg`) and add it to your system PATH.

## Acknowledgments

This project uses the **EnCodec** neural audio codec for feature extraction. Our wrapper extracts the latent space using an adaptation of the Hugging Face `transformers` port of the 24kHz model. 

* **Paper:** [High Fidelity Neural Audio Compression](https://openreview.net/pdf?id=ivCd8z8zR2) (Défossez et al., 2022)
* **Original Repository:** [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
* **Hugging Face Model:** [facebook/encodec_24khz](https://huggingface.co/facebook/encodec_24khz)

## 📖 Citation

If you use **Latent Secret Spin (LSS)** or this repository in your work, please cite our paper:

```bibtex
@inproceedings{coletta26_odyssey,
  title     = {Latent Secret Spin: Keyed Orthogonal Rotations for Blind Speech Watermarking in Anisotropic Latent Spaces},
  author    = {Emma Coletta and Massimiliano Todisco and Michele Panariello and Antonio Faonio and Nicholas Evans},
  booktitle = {Odyssey 2026},
  pages      = {35--41},
  year       = {2026},
  doi        = {10.21437/Odyssey.2026-6}
}
```

**Paper**

- DOI: `10.21437/Odyssey.2026-6`
