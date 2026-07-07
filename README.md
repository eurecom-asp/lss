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
* `LSS_embedding.py`: Embeds the secret payload into a batch of speech files.
* `apply_manipulations.py`: Applies signal distortions to test watermark robustness.
* `LSS_detection.py`: Scans speech files and calculates the watermark presence score.
