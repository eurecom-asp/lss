import os
import math
import json
import struct
import hmac
import hashlib
import random
import GPUtil
import yaml
import numpy as np
import argparse
from typing import List, Tuple, Dict, Optional, Any
from copy import deepcopy
from pathlib import Path
import pandas as pd

import torch
import torchaudio
import inspect

#from audio_processing import *


from datetime import datetime

# Encodec (NEW!)
#from encodec import EncodecModel
from transformers import EncodecModel




# Split

SPLIT_SEED   = 1337
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # by speakers

# ==========================================================================

def set_deterministic(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
#                              PRF utilities
# =============================================================================


def prf_bytes(key: bytes, info: bytes, nbytes: int) -> bytes:
    out = b''; ctr = 0
    while len(out) < nbytes:
        out += hmac.new(key, info + struct.pack('>I', ctr), hashlib.sha256).digest()
        ctr += 1
    return out[:nbytes]

def prf_gaussian_old(key: bytes, info: bytes, shape) -> torch.Tensor:
    s = int.from_bytes(prf_bytes(key, info, 32), 'big') % (2**63 - 1)
    gen = torch.Generator(device='cpu'); gen.manual_seed(s)
    return torch.randn(*shape, generator=gen)

def prf_gaussian(key: bytes, info: bytes, shape, device=None, dtype=torch.float32) -> torch.Tensor:
    s = int.from_bytes(prf_bytes(key, info, 32), 'big') % (2**63 - 1)

    gen = torch.Generator(device='cpu')
    gen.manual_seed(s)

    x = torch.randn(*shape, generator=gen, dtype=dtype)

    if device is not None:
        x = x.to(device)

    return x

def derive_wrong_key_hex(main_key: bytes) -> str:
    """Deterministically derive a different key from the main key (hex string)."""
    dk = hmac.new(main_key, b"wrong-key-derivation", hashlib.sha256).digest()[:16]
    return dk.hex()

# =============================================================================
#                              LINEAR ALGEBRA helpers
# =============================================================================

def householder_chain_Q(key: bytes, nonce: bytes, chunk_idx: int, n: int, m: int = 0, device='cpu') -> torch.Tensor:
    """Q ∈ SO(n) as product of m Householder reflectors; m=0 → Identity."""
    if m <= 0:
        return torch.eye(n, device=device)
    Q = torch.eye(n, device=device)
    for k in range(m):
        info = b'HH' + nonce + struct.pack('>II', chunk_idx, k)
        v = prf_gaussian(key, info, (n,)).to(device)
        v = v / (v.norm() + 1e-12)
        H = torch.eye(n, device=device) - 2.0 * torch.outer(v, v)
        Q = H @ Q
    if torch.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]
    return Q

def givens_apply_(Z: torch.Tensor, i: int, j: int, phi: float):
    """In-place Givens rotation on rows i,j across all columns of Z (n × T_enc)."""
    '''phi_t = torch.tensor(phi, dtype=Z.dtype, device=Z.device)
    c, s = torch.cos(phi_t), torch.sin(phi_t)
    Zi = Z[i, :].clone()
    Zj = Z[j, :].clone()

    Z[i, :] = c * Zi - s * Zj
    Z[j, :] = s * Zi + c * Zj'''


    c = torch.cos(torch.tensor(phi, device=Z.device, dtype=Z.dtype))
    s = torch.sin(torch.tensor(phi, device=Z.device, dtype=Z.dtype))

    Zi = Z[i]
    Zj = Z[j]

    tmp_i = c * Zi - s * Zj
    tmp_j = s * Zi + c * Zj

    Z[i] = tmp_i
    Z[j] = tmp_j

# =============================================================================
#                              PCA fit (center-only)
# =============================================================================


@torch.no_grad()
def pca_fit(latents_list: List[torch.Tensor], eps: float = 1e-8):
    """
    Fit PCA on list of matrices [n × T_k] (no whitening).
    Returns W=Uᵀ, Winv=U, mu, lam (eigenvalues sorted desc).
    """
    X = torch.cat(latents_list, dim=1)         # n × total_T
    mu = X.mean(dim=1, keepdim=True)           # n × 1
    Xc = X - mu
    cov = (Xc @ Xc.T) / max(1, Xc.shape[1])    # n × n
    evals, evecs = torch.linalg.eigh(cov)      # ascending
    idx = torch.argsort(evals, descending=True)
    lam = evals[idx].clamp_min(eps).contiguous()   # [n]
    U   = evecs[:, idx].contiguous()               # [n × n]
    W    = U.T                                     # Z = W (F - mu)
    Winv = U                                       # F ≈ Winv Z + mu
    return W, Winv, mu, lam


def chunk_ranges(T: int, M: int):
    s = 0
    while s < T:
        e = min(T, s + M)
        yield s, e
        s = e


# =========================================================================
#                         AUDIO PROCESSING HELPERS
# =========================================================================

def robust_resample_protocol(wav_24k, target_sr=16000):
    """
    Simulates the AudioSeal environment:
    Downsample to 16kHz (distribution format) -> Upsample to 24kHz (detection format).
    Supports Batch [B, 1, T].
    """
    # 24k -> 16k
    wav_16k = torchaudio.functional.resample(wav_24k, 24000, target_sr)
    # 16k -> 24k
    wav_24k_rec = torchaudio.functional.resample(wav_16k, target_sr, 24000)
    return wav_24k_rec



# =============================================================================
#                               PROTOCOL PARSING
# =============================================================================


def load_eval_subset(proto_tsv: str, base_flac: str, attack_label: str) -> List[Tuple[str, str, str]]:
    """
    Keep lines where:
      - 8th column == attack_label (e.g., 'A26' or 'A28')
      - 4th column == '-' (no codec)
    Return [(speaker, file_id, path)].
    """
    keep = []
    with open(proto_tsv, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            cols = line.split()
            if len(cols) < 8:
                continue
            speaker, file_id, col4, atk = cols[0], cols[1], cols[3], cols[7]
            if atk == attack_label and col4 == '-':
                flac = os.path.join(base_flac, f"{file_id}.flac")
                if os.path.isfile(flac):
                    keep.append((speaker, file_id, flac))
    return keep



def split_by_speaker(items: List[Tuple[str, str, str]],
                     ratios=(0.6, 0.2, 0.2),
                     seed=SPLIT_SEED):
    spk2items: Dict[str, List[Tuple[str, str, str]]] = {}
    for spk, fid, path in items:
        spk2items.setdefault(spk, []).append((spk, fid, path))
    speakers = sorted(spk2items.keys())
    rng = random.Random(seed); rng.shuffle(speakers)
    n = len(speakers)
    n_train = int(ratios[0] * n); n_dev = int(ratios[1] * n)
    train_spk = set(speakers[:n_train])
    dev_spk   = set(speakers[n_train:n_train + n_dev])
    test_spk  = set(speakers[n_train + n_dev:])
    def collect(sset):
        out = []
        for spk in sset: out.extend(spk2items[spk])
        return out
    return {'train': collect(train_spk), 'dev': collect(dev_spk), 'test': collect(test_spk)}


def load_bonafide(proto_tsv: str, base_flac: str):
    """
    Load only bonafide samples from the ASVspoof5 TSV.
    Returns: [(speaker, file_id, flac_path)]
    """
    keep = []
    with open(proto_tsv, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            cols = line.split()
            if len(cols) < 9:
                continue

            speaker = cols[0]
            file_id = cols[1]
            codec_label = cols[3]
            attack_label = cols[7]  # 'bonafide' for genuine samples

            if attack_label == "bonafide" and codec_label == '-':
                flac = os.path.join(base_flac, f"{file_id}.flac")
                if os.path.isfile(flac):
                    keep.append((speaker, file_id, flac))

    return keep


# ------------------------- PCA helpers --------------------------

def load_pca_stats(path, device):
    """
    Loads PCA statistics from a .pt file.
    Returns: (W, Winv, mu, lam) on the specified device.
    """
    print(f"[PCA] Loading state from: {path}")
    pca = torch.load(path, map_location=device)

    return (pca['W'].to(device),
            pca['Winv'].to(device),
            pca['mu'].to(device),
            pca['lam'].to(device))


def save_pca_stats(lsp, max_fit, config):
    """
    Generates a dynamic filename and saves the current PCA state.
    """
    # 1. Get Directory
    # Fallback to current dir if not set, but user provided a specific path in prompt
    save_dir = config['paths']['pca_save_dir']
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    # 2. Generate Filename: PCA_{date}_{time}_{seed}_{max_whiten}.pt
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    seed = config['split']['seed']

    #max_fit = config['script']['max_whiten']

    filename = f"PCA_{timestamp}_{seed}_{max_fit}.pt"
    save_path = os.path.join(save_dir, filename)

    # 3. Save
    print(f"[PCA] Saving state to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'W': lsp.W,
        'Winv': lsp.Winv,
        'mu': lsp.mu,
        'lam': lsp.lam
    }, save_path)

    print(f"[INFO] Saved! To reuse this later, set 'pca_state_path' to this file.")
    return save_path



# =============================================================================
#                              EER helper
# =============================================================================



def compute_eer(scores_pos: List[float], scores_neg: List[float]) -> Tuple[float, float]:
    """
    Optimized EER calculation using vectorization.
    Complexity: O(N log N) instead of O(N^2).
    """
    scores_pos = np.array(scores_pos)
    scores_neg = np.array(scores_neg)

    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return 0.5, 0.0

    # 1. Prepare Labels (1=Pos, 0=Neg) and Concatenate
    # We treat "Presence" as the positive class.
    y_true = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    y_scores = np.concatenate([scores_pos, scores_neg])

    # 2. Sort indices by score (Descending)
    # This allows us to sweep the threshold from High -> Low
    desc_sort_idxs = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_sort_idxs]
    y_scores_sorted = y_scores[desc_sort_idxs]

    # 3. Calculate Cumulative Sums (Vectorized TP and FP counts)
    # At index 'i', we treat everything from 0..i as "Predicted Positive"

    # Total Positive/Negative samples in the set
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)

    # TP[i] = Count of actual positives found so far (from top score down to i)
    tps = np.cumsum(y_true_sorted)

    # FP[i] = Count of actual negatives found so far
    # (Total items processed 'i+1') - (TPs found 'tps')
    fps = (np.arange(len(y_true_sorted)) + 1) - tps

    # 4. Calculate Rates
    # TPR = TP / Total_Pos
    # FPR = FP / Total_Neg
    tpr = tps / n_pos
    fpr = fps / n_neg

    # FNR = 1 - TPR
    fnr = 1 - tpr

    # 5. Find EER Index (where |FPR - FNR| is minimal)
    diffs = np.abs(fpr - fnr)
    min_idx = np.argmin(diffs)

    eer = (fpr[min_idx] + fnr[min_idx]) / 2
    threshold = y_scores_sorted[min_idx]

    return float(eer), float(threshold)

# =============================================================================
#                              SAVING SCORES LSS
# =============================================================================

def _get_reporting_filenames(
        config: Dict[str, Any],
        tsv_core_name: str,
        eval_count: int,
        key_variant: str
) -> Tuple[str, str, str]:
    """
    Generates the scenario name, **sub-directory name**, and core file names.
    Parses the PCA fit count directly from the filename in 'pca_load_path'.
    """
    manip_name = config['manipulation']['name']

    # 1. Determine Base Folder Name based on Manipulation
    if manip_name == 'clean':
        subdir_base_name = "clean"
    elif manip_name == 'lowpass_hz':
        cutoff_low = int(float(config['manipulation']['cutoff_low']))
        subdir_base_name = f'lowpass_{cutoff_low}Hz'
    elif manip_name == 'highpass_hz':
        cutoff_high = int(float(config['manipulation']['cutoff_high']))
        subdir_base_name = f'highpass_{cutoff_high}Hz'
    elif manip_name == 'bandpass_hz':
        c_low = int(float(config['manipulation']['cutoff_bp_low']))
        c_high = int(float(config['manipulation']['cutoff_bp_high']))
        subdir_base_name = f'bandpass_{c_low}_{c_high}Hz'
    elif manip_name == 'mp3_kbps':
        mp3_bitrate = int(float(config['manipulation']['mp3_bitrate']))
        subdir_base_name = f'mp3_{mp3_bitrate}kbps'
    elif manip_name == 'resampling':
        target_resample = int(float(config['manipulation']['target_resample']) / 1000)
        orig_freq = int(float(config['encodec']['sample_rate']) / 1000)
        subdir_base_name = f'resample_{orig_freq}_to_{target_resample}kHz'
    elif manip_name == 'white_noise':
        snr_db = int(float(config['manipulation']['snr_db']))
        subdir_base_name = f'white_noise_{snr_db}dB'
    elif manip_name == 'pink_noise':
        snr_db = int(float(config['manipulation']['snr_db']))
        subdir_base_name = f'pink_noise_{snr_db}dB'
    else:
        subdir_base_name = f'{manip_name}'  # Fallback

    # 2. Extract PCA Fit Count from Filename
    # Filename format expected: PCA_date_time_seed_COUNT.pt
    # Example: PCA_voxpopuli_20260127_1540_1337_50000.pt -> 50000

    pca_fit_count = config['script']['max_files']  # Default fallback

    if config['modes']['load_pca']:
        try:
            pca_path = config['paths']['pca_load_path']
            filename = os.path.basename(pca_path)  # -> PCA_..._50000.pt
            name_no_ext = os.path.splitext(filename)[0]  # -> PCA_..._50000
            extracted_samples_count = name_no_ext.split('_')[-1]  # -> 50000

            # Simple check to ensure it's a number
            if extracted_samples_count.isdigit():
                pca_fit_count = extracted_samples_count
        except Exception as e:
            print(f"[WARN] Could not parse PCA count from filename. Using config value. Error: {e}")

    # 3. Construct Sub-Directory Name
    # Format: {manip}_{seed}_{PCA_COUNT}_{max_eval}
    sub_dir_name = f"{subdir_base_name}_{config['split']['seed']}_{pca_fit_count}_{eval_count}"

    # 4. Construct Core Filename (tsv_core_name + key_variant)
    core_filename = f"{key_variant}{tsv_core_name}"

    # 5. Construct Scenario Name for Printing/JSON
    log_folder_name = os.path.basename(sub_dir_name)
    key_text = "WRONG-KEY" if "wrongkey_" in key_variant else "CORRECT-KEY"
    scenario_name = f"{tsv_core_name.split('_')[2].upper()}-{key_text}-{log_folder_name.upper()}"

    return scenario_name, sub_dir_name, core_filename


def save_and_report_eer(
        scores_pos: List[float],
        scores_neg: List[float],
        scores_save_dir: str,
        tsv_core_name: str,
        eval_count: int,
        config: Dict[str, Any],
        key_variant: str = ""  # correct -> "", wrong -> "_wrong"
) -> None:
    """
    Computes EER/Threshold, prints results, saves raw scores (TSV),
    and saves the EER/Threshold (JSON) for later use.
    """
    if not scores_pos or not scores_neg:
        print(f"[SKIP-Report] No scores for {tsv_core_name}{key_variant}.")
        return

    # --- A. FILENAME & SCENARIO NAME GENERATION ---
    scenario_name, sub_dir_name, core_filename = _get_reporting_filenames(
        config,
        tsv_core_name,
        eval_count,
        key_variant
    )

    def_save_dir = os.path.join(scores_save_dir, sub_dir_name)
    os.makedirs(def_save_dir, exist_ok=True)

    # --- B. DEBUG & EER CALCULATION ---
    mpos = sum(scores_pos) / len(scores_pos)
    mneg = sum(scores_neg) / len(scores_neg)

    print(f"[DEBUG-{scenario_name}] Mean scores: pos={mpos:.6e}, neg={mneg:.6e}, "
          f"pos>0={sum(s > 0 for s in scores_pos)}/{len(scores_pos)}, "
          f"neg>0={sum(s > 0 for s in scores_neg)}/{len(scores_neg)}")

    eer, tau = compute_eer(scores_pos, scores_neg)

    # --- C. PRINT RESULTS ---
    print(f"[RESULT-{scenario_name}] Presence EER = {100 * eer:.2f}% at threshold τ ≈ {tau:.6f}")
    print(f"[COUNTS-{scenario_name}] Positives={len(scores_pos)}, Negatives={len(scores_neg)}")

    # --- D. SAVE RAW SCORES (TSV) ---
    try:
        tsv_path = os.path.join(def_save_dir, f"{core_filename}.tsv")
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("set\tindex\tscore\n")
            for i, s in enumerate(scores_pos): f.write(f"pos\t{i}\t{s:.8f}\n")
            for i, s in enumerate(scores_neg): f.write(f"neg\t{i}\t{s:.8f}\n")
        print(f"[SAVE] Wrote scores to {tsv_path}")
    except Exception as e:
        print(f"[WARN] Could not write {tsv_path}: {e}")

    # --- E. SAVE SUMMARY METRICS (JSON) ---
    summary_data = {
        "scenario": scenario_name,
        "eer_score": float(eer),
        "eer_threshold": float(tau),  # <-- Threshold saved for later calculation of AUC/Accuracy
        "pos_count": len(scores_pos),
        "neg_count": len(scores_neg)
    }

    try:
        json_path = os.path.join(def_save_dir, f"{core_filename}_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4)
        print(f"[SAVE] Wrote summary metrics to {json_path}")
    except Exception as e:
        print(f"[WARN] Could not write {json_path}: {e}")


def save_pesq_results(results, metric_anchor, metric_total, file_path, config):

    print("\n" + "=" * 40)
    print(" LSS PERCEPTUAL QUALITY (PESQ-WB)")
    print("=" * 40)

    avg_anchor = metric_anchor.compute().item()
    avg_total = metric_total.compute().item()
    avg_tax = avg_anchor - avg_total

    print(f"Global Anchor (Codec):   {avg_anchor:.4f}")
    print(f"Global Total (Watermark):{avg_total:.4f}")
    print(f"Degradation (Tax):       {avg_anchor - avg_total:.4f}")

    # Save TSV
    #script_info = f"{config['split']['seed']}_{len(train_items)}_{len(eval_items)}"
    #tsv_path = os.path.join(scores_save_dir, 'pesq_scores', f"{manip_name}_{script_info}.tsv")
    pd.DataFrame(results).to_csv(file_path, sep='\t', index=False)
    print(f"[SAVE] PESQ scores saved to: {file_path}")
    print("=" * 40)

    # Save Global Summary to JSON
    # Creates a file like: pesq_scores/clean_1337_..._global.json
    global_stats = {
        "manipulation": config['manipulation']['name'],
        "global_anchor_pesq": float(avg_anchor),
        "global_total_pesq": float(avg_total),
        "global_tax": float(avg_tax),
        "total_files_evaluated": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    json_path = file_path.replace(".tsv", "_global.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(global_stats, f, indent=4)
        print(f"[SAVE] Global stats saved to: {json_path}")
    except Exception as e:
        print(f"[WARN] Failed to save Global JSON: {e}")

    print("=" * 40)

# =============================================================================
#                               GPU UTILITIES
# =============================================================================

def set_visible_gpus(gpu_ids: Optional[list[int]] = None, default_device: str = "cpu") -> list[torch.device]:
    """
    Return a list of torch.device objects corresponding to the specified physical GPU indices.
    If no GPUs are specified or CUDA is unavailable, returns CPU.

    Args:
        gpu_ids (List[int], optional): List of physical GPU indices. If None or empty, use CPU.
        default_device (str): Device to use if no GPUs are available.

    Returns:
        devices (List[torch.device]): List of torch.device objects for the selected GPUs (or CPU).
    """
    if gpu_ids is None or len(gpu_ids) == 0 or not torch.cuda.is_available():
        devices = [torch.device(default_device)]
        print(f"No GPUs specified or CUDA unavailable. Using device: {devices[0]}")
    else:
        devices = [torch.device(f"cuda:{i}") for i in gpu_ids]
        print(f"Using physical GPUs: {gpu_ids}")
        print(f"Devices to use in code: {devices}")

    return devices


def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for CPU-only)

    :param id: if specified, corresponds to the GPU index desired.
    """
    if id is None:
        # CPU only
        print('GPU not selected')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        # -1 for automatic choice
        device = id if id != -1 else GPUtil.getFirstAvailable(order='memory')[0]
        try:
            name = GPUtil.getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most available one.')
            device = GPUtil.getFirstAvailable(order='memory')[0]
            name = GPUtil.getGPUs()[device].name
        print('GPU selected: %d - %s' % (device, name))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # # Set memory growth
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    return device


# =============================================================================
#                         FILE HANDLING UTILITIES
# =============================================================================


def read_yaml(config_path):
    """
    Read YAML file.

    :param config_path: path to the YAML config file.
    :type config_path: str
    :return: dictionary correspondent to YAML content
    :rtype dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def discover_audio_paths(input_path: str) -> list:
    valid_exts = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.opus'}
    if not input_path:
        return []
    if os.path.isfile(input_path):
        return [input_path] if os.path.splitext(input_path)[1].lower() in valid_exts else []
    elif os.path.isdir(input_path):
        paths = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    paths.append(os.path.join(root, file))
        return paths
    return []
