# Latent Secret Spin (LSS)

### Keyed Orthogonal Rotations for Blind Speech Watermarking in Anisotropic Latent Spaces

Latent Secret Spin (LSS) is a latent-space speech watermarking framework designed for neural audio codecs. Instead of embedding information directly in the waveform, LSS modifies the codec's latent representations, enabling robust, minimally perceptible watermarking while remaining compatible with modern neural speech generation and compression pipelines.

![LSS Pipeline]()

The framework provides a complete pipeline for PCA estimation, watermark embedding, decoding, re-encoding, and watermark detection. It includes utilities for PCA-based latent modeling, secret-key generation, payload embedding, and both latent-domain and waveform-domain detection. LSS supports reproducible research and adapts to different neural codec architectures with minimal changes.

## 💡 Theory Principles and Implementation

Traditional watermarking systems modify the raw speech waveform or frequency spectrum. In contrast, LSS operates on the geometric structure of the latent representation rather than the waveform itself. This design improves robustness to common signal degradations while minimizing perceptual quality degradation. The framework also requires no neural network retraining.

The algorithm relies on four core stages:

### 1. Latent Extraction

The LSS framework passes raw speech through a pre-trained neural encoder (EnCodec). The encoder compresses the speech into continuous latent feature maps.

### 2. PCA Projection

The system projects these latent features into a Principal Component Analysis (PCA) space. This coordinate system decorrelates the features and isolates the variance. The resulting space is anisotropic: different principal components contain different amounts of signal energy. LSS exploits this anisotropy to identify suitable embedding directions for watermark insertion within the PCA domain.

### 3. Keyed Rotation (Embedding)

The embedding process uses a secret cryptographic key to generate a pseudo-random schedule. This schedule selects specific pairs of principal components (geometric planes) across different time chunks.

The system applies small orthogonal rotations (Givens rotations) to these planes to encode the binary payload. Rotating two principal axes with different variances introduces a measurable cross-covariance between them. This creates a latent signature that can later be recovered during detection.

For small rotation angles, the induced covariance is approximately proportional to the rotation angle. Consequently, its sign reveals the direction of the applied rotation and serves as the fundamental watermark carrier.

### 4. Blind Detection

The detector does not require access to the original speech recording. Instead, it extracts latent features from a suspect audio file and reconstructs the embedding schedule using the secret key.

The system isolates the targeted geometric planes and computes their cross-covariance values. If the correct key is supplied, the extracted covariance terms sum constructively and produce a strong detection score. If the key is incorrect, or if the audio does not contain a watermark, the covariance contributions tend to cancel out and yield a score close to zero.

This repository contains the reference implementation of LSS together with the evaluation protocol described in the accompanying paper.

## 🗂️ Repository Structure

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
