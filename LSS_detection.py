#!/usr/bin/env python3
import os
import math
import argparse
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils import *
from src.encodec_wrapper import EncodecLatent_NEW
from src.lss_pca import LatentSpinPCA
from src.pesq_eval import PesqMetricTorch
from src.audio_processing import apply_manipulation_batch

# --- Custom Paired Dataset for Post-Embedding Evaluation ---
class PairedAudioDataset(Dataset):
    def __init__(self, target_paths, ref_paths_dict, target_sr, duration=10.0):
        self.target_paths = target_paths
        self.ref_paths_dict = ref_paths_dict
        self.target_sr = target_sr
        self.target_samples = int(target_sr * duration)

    def __len__(self):
        return len(self.target_paths)

    def _load_and_format(self, path):
        wav, sr = torchaudio.load(str(path))
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        current_samples = wav.shape[-1]
        if current_samples < self.target_samples:
            n_repeats = (self.target_samples // current_samples) + 1
            wav = wav.repeat(1, n_repeats)[:, :self.target_samples]
        elif current_samples > self.target_samples:
            wav = wav[:, :self.target_samples]

        return wav

    def __getitem__(self, idx):
        target_path = self.target_paths[idx]
        fid = Path(target_path).stem

        wav_target = self._load_and_format(target_path)

        wav_ref = torch.zeros_like(wav_target)
        has_ref = False

        if self.ref_paths_dict and fid in self.ref_paths_dict:
            wav_ref = self._load_and_format(self.ref_paths_dict[fid])
            has_ref = True

        return wav_target, wav_ref, fid, has_ref


def custom_collate_paired(batch):
    wavs_target, wavs_ref, ids, has_refs = zip(*batch)
    return torch.stack(wavs_target), torch.stack(wavs_ref), list(ids), list(has_refs)


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="LSS Post-Embedding Detection & Evaluation Tool")
    parser.add_argument("-i", "--input", help="Directory of suspect/watermarked audio files (overrides config)")
    parser.add_argument("-c", "--clean_reference", help="Directory of clean original audio files (Enables EER & PESQ)")
    parser.add_argument("-r", "--results_dir", help="Directory path to save evaluation results (overrides config)")
    parser.add_argument("-d", "--device", type=int, help="Physical GPU device ID (overrides config file)")
    parser.add_argument("--config", default="config/config_LSS.yaml", help="Path to configuration file")
    parser.add_argument("--pca_load_dir", help="Directory path for estimated PCA basis (overrides config)")
    parser.add_argument("--key", type=str, help="Correct secret key sequence string (overrides config)")
    parser.add_argument("--wrong_key", help="Wrong secret key sequence string for evaluation (overrides config)")
    parser.add_argument("--wrong_key_eval", choices=['true', 'false'], help="Evaluate wrong secret key sequence (overrides config)")
    parser.add_argument("--theta", type=float, help="Rotation angle magnitude [radians] (overrides config)")
    parser.add_argument("--bits", help="Payload bit string (overrides config)")
    parser.add_argument("--threshold", type=float, help="Manual decision threshold (overrides EER calculation)")
    parser.add_argument("--compute_pesq", choices=['true', 'false'], help="Enable PESQ computation (overrides config)")

    # On-the-fly manipulation arguments
    parser.add_argument("--attack",
                        choices=['clean', 'lowpass_hz', 'highpass_hz', 'bandpass_hz', 'mp3_kbps', 'resampling',
                                 'white_noise', 'pink_noise'],
                        help="Apply a signal manipulation attack directly before feature extraction")
    parser.add_argument("--manip_param", type=float, nargs='+',
                        help="Override the core value for the selected attack (e.g., cutoff Hz, bitrate, SNR dB)")
    parser.add_argument("--save_manipulated", choices=['true', 'false'],
                        help="Save the manipulated audio tracks to disk (overrides config)")
    args = parser.parse_args()

    # Read configuration
    config = read_yaml(args.config)

    # ---------------------------------------------------------
    # 1. PARAMETER RESOLUTION (CLI overrides Config)
    # ---------------------------------------------------------

    # Device
    target_device = args.device if args.device is not None else config.get('device', -1)
    set_gpu(target_device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Paths
    input_target = args.input or config['paths']['detect_audio_dir']
    clean_target = args.clean_reference or config['paths']['clean_audio_dir']
    results_dir = args.results_dir or config['paths']['results_dir']
    pca_path = args.pca_load_dir or config['paths']['pca_load_dir']

    # Embedding Parameters
    correct_key = args.key if args.key is not None else config['run_lss']['key']
    wrong_key = args.wrong_key if args.wrong_key is not None else config['run_lss']['wrong_key']
    theta = args.theta if args.theta is not None else config['pca_lss']['theta']

    bit_str = args.bits or config['script']['bits'][:config['script']['payload_length']]
    bits = [1 if c == '1' else 0 for c in bit_str if c in '01']

    # Boolean Evaluation Modes
    if args.wrong_key_eval is not None:
        wrong_key_eval = args.wrong_key_eval == 'true'
    else:
        wrong_key_eval = config['modes']['wrong_key_eval']

    if args.compute_pesq is not None:
        compute_pesq = args.compute_pesq == 'true'
    else:
        compute_pesq = config['run_lss']['compute_pesq']

    if args.save_manipulated is not None:
        config['run_lss']['save_manipulated_wav'] = (args.save_manipulated == 'true')

    # ---------------------------------------------------------
    # 2. DYNAMIC ATTACK CONFIGURATION
    # ---------------------------------------------------------
    if args.attack:
        config['manipulation']['name'] = args.attack
    manip_name = config['manipulation']['name']

    if args.manip_param is not None:
        # Extract the primary parameter
        p1 = args.manip_param[0]

        if manip_name == 'lowpass_hz':
            config['manipulation']['cutoff_low'] = p1
        elif manip_name == 'highpass_hz':
            config['manipulation']['cutoff_high'] = p1
        elif manip_name == 'bandpass_hz':
            config['manipulation']['cutoff_bp_low'] = p1
            # Apply the second parameter if the user provided it
            if len(args.manip_param) > 1:
                config['manipulation']['cutoff_bp_high'] = args.manip_param[1]
        elif manip_name == 'mp3_kbps':
            config['manipulation']['mp3_bitrate'] = int(p1)
        elif manip_name == 'resampling':
            config['manipulation']['target_resample'] = int(p1)
        elif manip_name in ['white_noise', 'pink_noise']:
            config['manipulation']['snr_db'] = p1


    # Generate dynamic naming for output files
    if manip_name == 'lowpass_hz':
        m_dir = f"lowpass_{int(config['manipulation']['cutoff_low'])}Hz"
    elif manip_name == 'highpass_hz':
        m_dir = f"highpass_{int(config['manipulation']['cutoff_high'])}Hz"
    elif manip_name == 'bandpass_hz':
        m_dir = f"bandpass_{int(config['manipulation']['cutoff_bp_low'])}_{int(config['manipulation']['cutoff_bp_high'])}Hz"
    elif manip_name == 'mp3_kbps':
        m_dir = f"mp3_{config['manipulation']['mp3_bitrate']}kbps"
    elif manip_name == 'resampling':
        target_khz = int(int(config['manipulation']['target_resample']) / 1000)
        orig_khz = int(config['encodec']['sample_rate'] / 1000)
        m_dir = f"resample_{orig_khz}_to_{target_khz}kHz"
    elif manip_name in ['white_noise', 'pink_noise']:
        m_dir = f"{manip_name}_{config['manipulation']['snr_db']}dB"
    else:
        m_dir = manip_name

    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 3. DISCOVER AND ALIGN FILES
    # ---------------------------------------------------------
    if not input_target:
        print("[FATAL] Missing target input directory. Provide via CLI or config.")
        return

    target_files = discover_audio_paths(input_target)
    if not target_files:
        print(f"[FATAL] No valid audio assets located at: {input_target}")
        return

    ref_paths_dict = {}
    if clean_target:
        ref_files = discover_audio_paths(clean_target)
        ref_paths_dict = {Path(p).stem: p for p in ref_files}
        print(f"[INFO] Evaluation Mode: Found {len(ref_paths_dict)} clean reference files for metric computation.")
    else:
        print("[INFO] Blind Scan Mode: No reference directory provided. Computing presence scores only.")

    # ---------------------------------------------------------
    # 4. INITIALIZE MODELS
    # ---------------------------------------------------------
    print("[INFO] Initializing EnCodec and PCA Latent Geometry...")
    enc = EncodecLatent_NEW(device=device)
    lsp = LatentSpinPCA(
        key_hex=correct_key, theta=theta, P_payload=config['pca_lss']['payload_planes'],
        P_pilots=config['pca_lss']['pilots'], M=config['pca_lss']['m_frames'], L=config['pca_lss']['L'],
        mixers=config['pca_lss']['householder_mixers'], repetition=config['pca_lss']['repetition_default'],
        device=device
    )

    try:
        W, Winv, mu, lam = load_pca_stats(pca_path, device)
        lsp.W, lsp.Winv, lsp.mu, lsp.lam = W, Winv, mu, lam
    except Exception as e:
        print(f"[WARN] Invalid or missing PCA basis at '{pca_path}'. Error: {e}")
        print("[FATAL] Detection requires a valid PCA basis. Exiting script.")
        return

    lsp_wrong = None
    if wrong_key_eval and clean_target:
        if not wrong_key:
            # Replicate the exact hex-check logic
            main_key_bytes = (bytes.fromhex(correct_key)
                              if all(c in '0123456789abcdef' for c in correct_key.lower())
                              else correct_key.encode())
            wrong_key = derive_wrong_key_hex(main_key_bytes)

        lsp_wrong = LatentSpinPCA(
            key_hex=wrong_key, theta=theta, P_payload=config['pca_lss']['payload_planes'],
            P_pilots=config['pca_lss']['pilots'], M=config['pca_lss']['m_frames'], L=config['pca_lss']['L'],
            mixers=config['pca_lss']['householder_mixers'], repetition=config['pca_lss']['repetition_default'],
            device=device
        )
        lsp_wrong.load_pca_from(lsp)

    # ---------------------------------------------------------
    # 5. DATASET & METRIC STORAGE
    # ---------------------------------------------------------
    dataset = PairedAudioDataset(target_files, ref_paths_dict, target_sr=enc.enc_samplerate, duration=10.0)
    loader = DataLoader(
        dataset, batch_size=config['batching']['test_batch_size'],
        shuffle=False, collate_fn=custom_collate_paired, num_workers=config['batching']['num_workers']
    )

    scores_pos, scores_neg = [], []
    scores_pos_w, scores_neg_w = [], []
    pesq_results, file_results = [], []

    metric_anchor = PesqMetricTorch(sample_rate=enc.enc_samplerate).to(device) if compute_pesq else None
    metric_total = PesqMetricTorch(sample_rate=enc.enc_samplerate).to(device) if compute_pesq else None

    # ---------------------------------------------------------
    # 6. PROCESSING LOOP
    # ---------------------------------------------------------
    print(f"\n--- Scanning {len(target_files)} Tracks ---")
    if manip_name != "clean":
        print(f"[TEST MODE] Applying manipulation attack: '{m_dir}' on-the-fly.")

    with torch.no_grad():
        for wavs_target, wavs_ref, batch_ids, has_refs in tqdm(loader, desc="Extracting Signatures", leave=False):
            wavs_target = wavs_target.to(device)
            wavs_ref = wavs_ref.to(device)
            B = wavs_target.shape[0]

            if manip_name != "clean":
                manip_out_dir = config['paths']['audio_wm_out_dir_manip'] if config['run_lss'].get(
                    'save_manipulated_wav') else None
                wavs_target = apply_manipulation_batch(wavs_target, config, out_dir=manip_out_dir, batch_ids=batch_ids,
                                                       target_sr=enc.enc_samplerate)

            if clean_target and compute_pesq:
                F_ref = enc.encode_test_batch(wavs_ref)
                clean_codec_audio = enc.decode_from_features(F_ref)
                scores_anc = metric_anchor.score_batch(clean_codec_audio, wavs_ref)
                scores_tot = metric_total.score_batch(wavs_target, wavs_ref)

            for i in range(B):
                fid = batch_ids[i]

                s_pos = lsp.presence_score_from_waveform(enc, wavs_target[i].unsqueeze(0), fid, bits)
                scores_pos.append(s_pos)
                file_results.append({"fid": fid, "score": s_pos})

                if has_refs[i]:
                    s_neg = lsp.presence_score_from_waveform(enc, wavs_ref[i].unsqueeze(0), fid, bits)
                    scores_neg.append(s_neg)

                    if lsp_wrong:
                        scores_pos_w.append(
                            lsp_wrong.presence_score_from_waveform(enc, wavs_target[i].unsqueeze(0), fid, bits))
                        scores_neg_w.append(
                            lsp_wrong.presence_score_from_waveform(enc, wavs_ref[i].unsqueeze(0), fid, bits))

                    if compute_pesq and not math.isnan(scores_anc[i]) and not math.isnan(scores_tot[i]):
                        pesq_results.append({'file_id': fid, 'pesq_anchor': scores_anc[i], 'pesq_total': scores_tot[i],
                                             'lss_tax': scores_anc[i] - scores_tot[i]})

    # ---------------------------------------------------------
    # 7. METRIC COMPUTATION & CLASSIFICATION
    # ---------------------------------------------------------
    eval_count = len(target_files)
    final_threshold = args.threshold

    if clean_target and scores_neg:
        print("\n--- Computing Benchmarks ---")
        from src.utils import compute_eer
        eer, tau = compute_eer(scores_pos, scores_neg)

        if final_threshold is None:
            final_threshold = tau
            print(f"[INFO] Automatically adopting optimal EER threshold: {final_threshold:.6f}")

        save_and_report_eer(scores_pos, scores_neg, results_dir, f"scores_eval_waveform", eval_count,
                            config, "correctkey_")

        if lsp_wrong:
            save_and_report_eer(scores_pos_w, scores_neg_w, results_dir, f"scores_eval_waveform",
                                eval_count, config, "wrongkey_")

        if compute_pesq and pesq_results:
            pesq_dir = os.path.join(results_dir, 'pesq_scores')
            os.makedirs(pesq_dir, exist_ok=True)
            tsv_path = os.path.join(pesq_dir, f"{m_dir}_{config['split']['seed']}_{eval_count}.tsv")
            save_pesq_results(pesq_results, metric_anchor, metric_total, tsv_path, config)

    elif final_threshold is None:
        print("\n[WARN] No reference directory or manual threshold provided.")
        print("[WARN] Classification defaults to 0.0. Pass --threshold to set a specific boundary.")
        final_threshold = 0.0

    print("\n" + "=" * 80)
    print(f"{'Target Asset Filename':<40} | {'Score':<15} | {'Classification':<15}")
    print("=" * 80)

    for res in file_results:
        classification = "WATERMARKED" if res['score'] >= final_threshold else "CLEAN"
        print(f"{res['fid'] + '.wav':<40} | {res['score']:>15.6f} | {classification:<15}")

if __name__ == "__main__":
    main()