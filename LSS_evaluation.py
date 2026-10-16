import os
import math
import argparse
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import *
from src.encodec_wrapper import EncodecLatent_NEW
from src.lss_pca import LatentSpinPCA
from src.audio_processing import *
from src.pesq_eval import *


def main():
    parser = argparse.ArgumentParser(description="LSS End-to-End Evaluation & Benchmarking Tool")
    parser.add_argument("-i", "--input", help="Directory of clean audio files for evaluation")
    parser.add_argument("--config", default="config/config_LSS.yaml", help="Path to configuration file")
    parser.add_argument("--attack",
                        choices=['clean', 'lowpass_hz', 'highpass_hz', 'bandpass_hz', 'mp3_kbps', 'resampling',
                                 'white_noise', 'pink_noise'],
                        help="Override the type of signal distortion technique applied")
    parser.add_argument("--param", type=float, help="Override the core channel value (e.g. cutoff Hz, bitrate, SNR dB)")
    parser.add_argument("--max_eval", type=int, help="Maximum number of files to evaluate")
    args = parser.parse_args()

    config = read_yaml(args.config)
    set_gpu(config['device'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Directory Setup ---
    input_target = args.input or config['paths'].get('embed_audio_dir')
    pca_path = config['paths']['pca_load_dir']
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # --- Handle dynamic configuration updates ---
    if args.attack:
        config['manipulation']['name'] = args.attack
    manip_name = config['manipulation']['name']

    if args.param is not None:
        if manip_name == 'lowpass_hz':
            config['manipulation']['cutoff_low'] = args.param
        elif manip_name == 'highpass_hz':
            config['manipulation']['cutoff_high'] = args.param
        elif manip_name == 'mp3_kbps':
            config['manipulation']['mp3_bitrate'] = int(args.param)
        elif manip_name == 'resampling':
            config['manipulation']['target_resample'] = int(args.param)
        elif manip_name in ['white_noise', 'pink_noise']:
            config['manipulation']['snr_db'] = args.param

    # --- Mode Flags ---
    latent_eval = config['modes'].get('latent_eval', False)
    both_eval = config['modes'].get('both_eval', True)
    wrong_key_eval = config['modes'].get('wrong_key_eval', True)
    compute_pesq = config['run_lss'].get('compute_pesq', True)

    # --- Discover Files ---
    files = discover_audio_paths(input_target)
    if not files:
        print(f"[FATAL] No valid clean audio assets located at: {input_target}")
        return

    max_eval = args.max_eval or config['script'].get('max_eval', len(files))
    if len(files) > max_eval:
        import random
        random.seed(config['split']['seed'])
        files = random.sample(files, max_eval)

    # --- Keys & Payload ---
    correct_key = config['run_lss']['key']
    bit_str = config['script']['bits'][:config['script']['payload_length']]
    bits = [1 if c == '1' else 0 for c in bit_str if c in '01']

    # --- Initialize Models ---
    print("[INFO] Loading Models and PCA Stats...")
    enc = EncodecLatent_NEW(device=device)

    lsp = LatentSpinPCA(
        key_hex=correct_key, theta=config['pca_lss']['theta'], P_payload=config['pca_lss']['payload_planes'],
        P_pilots=config['pca_lss']['pilots'], M=config['pca_lss']['m_frames'], L=config['pca_lss']['L'],
        mixers=config['pca_lss']['householder_mixers'], repetition=config['pca_lss']['repetition_default'],
        device=device
    )

    W, Winv, mu, lam = load_pca_stats(pca_path, device)
    lsp.W, lsp.Winv, lsp.mu, lsp.lam = W, Winv, mu, lam

    lsp_wrong = None
    if wrong_key_eval:
        wrong_hex = derive_wrong_key_hex(correct_key.encode())
        lsp_wrong = LatentSpinPCA(
            key_hex=wrong_hex, theta=config['pca_lss']['theta'], P_payload=config['pca_lss']['payload_planes'],
            P_pilots=config['pca_lss']['pilots'], M=config['pca_lss']['m_frames'], L=config['pca_lss']['L'],
            mixers=config['pca_lss']['householder_mixers'], repetition=config['pca_lss']['repetition_default'],
            device=device
        )
        lsp_wrong.load_pca_from(lsp)

    # --- Dataset Setup ---
    dataset = AudioDataset(file_paths=files, target_sr=enc.enc_samplerate, duration=10.0, mode='test')
    loader = DataLoader(dataset, batch_size=config['batching']['test_batch_size'], shuffle=False,
                        collate_fn=custom_collate_asv, num_workers=config['batching']['num_workers'])

    # --- Score Trackers ---
    scores_pos_lat, scores_neg_lat = [], []
    scores_pos_lat_w, scores_neg_lat_w = [], []
    scores_pos_wave, scores_neg_wave = [], []
    scores_pos_wave_w, scores_neg_wave_w = [], []
    scores_pos_wave_m, scores_neg_wave_m = [], []
    scores_pos_wave_w_m, scores_neg_wave_w_m = [], []

    metric_anchor = PesqMetricTorch(sample_rate=enc.enc_samplerate).to(device) if compute_pesq else None
    metric_total = PesqMetricTorch(sample_rate=enc.enc_samplerate).to(device) if compute_pesq else None
    pesq_results = []

    print(f"\n[EVAL] Starting Benchmark Pipeline on {len(files)} tracks...")
    print(f"[EVAL] Attack Mode: {manip_name}")

    with torch.no_grad():
        for batch_wavs, batch_ids in tqdm(loader, desc="Benchmarking Progress"):
            if batch_wavs is None: continue
            batch_wavs = batch_wavs.to(device)
            B = batch_wavs.shape[0]

            # 1. Encode & Embed
            F_orig = enc.encode_test_batch(batch_wavs)
            F_wm = lsp.embed_features_batch(F_orig, batch_ids, bits)

            # 2. Latent Detection
            if latent_eval or both_eval:
                for i in range(B):
                    fid = batch_ids[i]
                    scores_pos_lat.append(lsp.presence_score_from_features(F_wm[i], fid, bits))
                    scores_neg_lat.append(lsp.presence_score_from_features(F_orig[i], fid, bits))
                    if lsp_wrong:
                        scores_pos_lat_w.append(lsp_wrong.presence_score_from_features(F_wm[i], fid, bits))
                        scores_neg_lat_w.append(lsp_wrong.presence_score_from_features(F_orig[i], fid, bits))

            # 3. Waveform Generation & Evaluation
            if (not latent_eval) or both_eval:
                wm_audio = enc.decode_from_features(F_wm)
                clean_codec_audio = enc.decode_from_features(F_orig)

                if compute_pesq:
                    scores_anc = metric_anchor.score_batch(clean_codec_audio, batch_wavs)
                    scores_tot = metric_total.score_batch(wm_audio, batch_wavs)

                wm_audio_manip = None
                if manip_name != 'clean':
                    wm_audio_manip = apply_manipulation_batch(wm_audio, config, out_dir=None, batch_ids=batch_ids,
                                                              target_sr=enc.enc_samplerate)

                for i in range(B):
                    fid = batch_ids[i]

                    if compute_pesq:
                        if not math.isnan(scores_anc[i]) and not math.isnan(scores_tot[i]):
                            pesq_results.append(
                                {'file_id': fid, 'pesq_anchor': scores_anc[i], 'pesq_total': scores_tot[i],
                                 'lss_tax': scores_anc[i] - scores_tot[i]})

                    # Clean Waveform Scores
                    scores_pos_wave.append(lsp.presence_score_from_waveform(enc, wm_audio[i], fid, bits))
                    scores_neg_wave.append(lsp.presence_score_from_waveform(enc, batch_wavs[i], fid, bits))

                    # Manipulated Waveform Scores
                    if manip_name != 'clean':
                        scores_pos_wave_m.append(lsp.presence_score_from_waveform(enc, wm_audio_manip[i], fid, bits))
                        scores_neg_wave_m.append(scores_neg_wave[-1])  # Compare against clean original

                    # Wrong Key Scores
                    if lsp_wrong:
                        scores_pos_wave_w.append(lsp_wrong.presence_score_from_waveform(enc, wm_audio[i], fid, bits))
                        scores_neg_wave_w.append(lsp_wrong.presence_score_from_waveform(enc, batch_wavs[i], fid, bits))
                        if manip_name != 'clean':
                            scores_pos_wave_w_m.append(
                                lsp_wrong.presence_score_from_waveform(enc, wm_audio_manip[i], fid, bits))
                            scores_neg_wave_w_m.append(scores_neg_wave_w[-1])

    print("\n--- Generating Benchmark Reports ---")
    eval_count = len(files)

    # Report execution using standard utils format
    if scores_pos_lat: save_and_report_eer(scores_pos_lat, scores_neg_lat, results_dir, "scores_eval_clean_latent",
                                           eval_count, config, "correctkey_")
    if scores_pos_wave: save_and_report_eer(scores_pos_wave, scores_neg_wave, results_dir, "scores_eval_clean_waveform",
                                            eval_count, config, "correctkey_")
    if scores_pos_wave_m: save_and_report_eer(scores_pos_wave_m, scores_neg_wave_m, results_dir,
                                              "scores_eval_manip_waveform", eval_count, config, "correctkey_")

    if lsp_wrong:
        if scores_pos_lat_w: save_and_report_eer(scores_pos_lat_w, scores_neg_lat_w, results_dir,
                                                 "scores_eval_clean_latent", eval_count, config, "wrongkey_")
        if scores_pos_wave_w: save_and_report_eer(scores_pos_wave_w, scores_neg_wave_w, results_dir,
                                                  "scores_eval_clean_waveform", eval_count, config, "wrongkey_")
        if scores_pos_wave_w_m: save_and_report_eer(scores_pos_wave_w_m, scores_neg_wave_w_m, results_dir,
                                                    "scores_eval_manip_waveform", eval_count, config, "wrongkey_")

    if compute_pesq and pesq_results:
        pesq_dir = os.path.join(results_dir, 'pesq_scores')
        os.makedirs(pesq_dir, exist_ok=True)
        tsv_path = os.path.join(pesq_dir, f"{manip_name}_{config['split']['seed']}_{eval_count}.tsv")
        save_pesq_results(pesq_results, metric_anchor, metric_total, tsv_path, config)


if __name__ == "__main__":
    main()