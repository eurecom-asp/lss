# Latent Secret Spin (LSS)
###  Keyed Orthogonal Rotations for Blind Speech Watermarking in Anisotropic Latent Spaces

Latent Secret Spin (LSS) is a latent-space speech watermarking framework designed for neural audio codecs. Instead of embedding information directly in the waveform, LSS modifies the codec's latent representations, enabling robust, minimally perceptible watermarking while remaining compatible with modern neural speech generation and compression pipelines.

![LSS Pipeline]() 

The framework provides a complete pipeline for PCA estimation, watermark embedding, decoding, re-encoding, and watermark detection. It includes utilities for PCA-based latent modeling, secret-key generation, payload embedding, and both latent-domain and waveform-domain detection. LSS supports reproducible research and adapts to different neural codec architectures with minimal changes.

## Theory and Operation

Standard watermarking modifies the raw speech waveform or frequency spectrum. In contrast, LSS operates on the geometric representation of the utterance signal, making the watermark inherently robust to signal degradation and helps preventing perceptual quality loss. The framework does not require any kind of neural network (re)training. 

The algorithm relies on four core stages:

### 1. Latent Extraction
The LSS framework passes raw speech through a pre-trained neural encoder (EnCodec). The encoder compresses the speech into continuous latent feature maps. 

### 2. PCA Projection
The system projects these latent features into a Principal Component Analysis (PCA) space. This coordinate system decorrelates the features and isolates the variance. The resulting space is anisotropic. Different principal components contain different amounts of signal energy. The framework exploits this uneven energy distribution to identify suitable embedding directions within the latent space, therefore exploiting the anisotropy to perform watermark embedding in the PCA domain.

### 3. Keyed Rotation (Embedding)
The embedding process uses a secret cryptographic key to generate a pseudo-random schedule. This schedule selects specific pairs of principal components (geometric planes) across different time chunks. 

The system applies small orthogonal rotations (Givens rotations) to these planes to encode the binary payload. Rotating two axes with different variances forces them to correlate. This creates a cross-covariance signature. 
For small angles, this term is approximately proportional to the rotation angle. Therefore, the sign of the covariance tells us the sign (or rather, direction) of the rotation. LSS uses this sign as the basic watermark carrier. 

### 4. Blind Detection
The detector does not need the original speech recording to verify the watermark. It extracts the latent features of a suspect file and applies the secret key to reconstruct the embedding schedule. 

The system isolates the targeted geometric planes and calculates their cross-covariance. If the secret key matches the original schedule, the extracted covariance terms sum constructively and yield a high detection score. If the key is incorrect, or the file lacks a watermark, the terms cancel out and yield a score near zero.

This repository contains the reference implementation of LSS and evaluation protocol proposed in the original paper. 

## Repository Structure

* `config/config_LSS.yaml`: The central configuration file controlling paths, payload bits, keys, and the (many) model configurable parameters.
* `src/`: Contains the core system modules (`audio_processing.py`, `encodec_wrapper.py`, `lss_pca.py`, `utils.py`).
* `estimate_pca.py`: Estimates the PCA basis from a dataset of clean speech.
* `LSS_embedding.py`: Embeds the secret payload into the selected speech audio samples.
* `apply_manipulations.py`: Applies signal distortions to test watermark robustness [optional].
* `LSS_detection.py`: Scans selected speech audio samples and calculates the watermark presence score.
* `LSS_evaluation.py`

## Setup and Configuration

Configure your base settings in `config/config_LSS.yaml` before running the tools. 

You must define:
* `paths.pca_estim_audio`: Target folder for your PCA training data.
* `run_lss.key`: Your secret cryptographic hex key (minimum 16 bytes).
* `script.bits`: The binary sequence to embed (e.g., `"10011101"`).

The tools operate as standalone scripts. You can override configuration defaults using terminal arguments.

## Quick Start

```bash
git clone ...
cd LSS
pip install -r requirements.txt
python estimate_pca.py
python LSS_embedding.py
python LSS_detection.py
```

### 1. Estimate the PCA Basis
You must estimate the principal components of your speech distribution before embedding. This builds the geometric space the system uses for rotations.

```bash
python estimate_pca.py -i /path/to/raw_dataset/ -o ./checkpoints/ --max_files 25000
```
 
### 2. Embed the LSS watermark
Run the embedding script to inject your payload into a target folder of speech files

```bash
python LSS_embedding.py -i /path/to/input_speech/ -o ./path/to/watermarked_output/
```
You can override the secret key, payload bits, temporal framing or other configurable parameters using terminal arguments, e.g.

```bash
python LSS_embedding.py -i ./path/to/input_speech/ -o ./path/to/watermarked_output/ --key "0123456789abcdef" --bits "11110000"
```
### 3. Apply DSP Manipulations
To test robustness of your waterark, we also make available a script that allows you to simulate standard signal distortions to watermarked speech. The script organises outputs into subfolders based on the non-malicious attack type. Supported attacks: `lowpass_hz`, `highpass_hz`, `bandpass_hz`, `mp3_kbps`, `resampling`, `white_noise`, `pink_noise` (see LSS paper). 

```bash
python apply_manipulations.py -i ./watermarked_output/ -o ./attacked_speech/ --attack mp3_kbps --param 128 --save_manipulated_wav
```
### 4. Detect the Watermark
The detection script operates in two distinct modes. You can run a full laboratory evaluation or a blind field scan.

**Laboratory Mode (Benchmarking)**
Provide both the suspect target speech and the corresponding clean original audio directories. The script calculates the Equal Error Rate (EER) and the Perceptual Evaluation of Speech Quality (PESQ) metric. It determines the optimal decision threshold automatically, saves the score reports to your results directory, and prints a classification table to your terminal.

```bash
python LSS_detection.py -i ./path/to/input_suspect_watermarked_speech -r ./path/to/corresponding_clean_speech/
```
**Field Mode (Blind Detection)**
Provide a folder of suspect speech and a known decision threshold. Do not provide a reference directory. The script extracts the latent signatures and prints a binary classification table to your terminal.

You must define this threshold based on your specific audio dataset and operational environment. **Do not set the threshold to zero.** Unwatermarked speech produces a range of scores centered near zero, including small positive numbers. A threshold of zero will flag clean files as watermarked.
To find the optimal threshold, run the Laboratory Mode evaluation test on a sample dataset containing both clean and watermarked files. The script calculates the Equal Error Rate (EER), which identifies the mathematical gap between the clean score distribution and the watermarked score distribution. Use this EER value as your baseline threshold for Field Mode.

You can adjust the threshold to dictate system behavior:

* **Raise the threshold to prioritize security**. This minimizes false positives but may miss watermarks degraded by severe audio distortion.
* **Lower the threshold to prioritize sensitivity**. This detects highly degraded watermarks but increases the risk of flagging clean files.

**On-the-fly manipulation**
