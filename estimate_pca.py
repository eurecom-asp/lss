import os
import argparse
import random
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import *
from src.encodec_wrapper import EncodecLatent_NEW
from src.lss_pca import LatentSpinPCA
from src.audio_processing import AudioDataset, custom_collate_asv



def main():
    parser = argparse.ArgumentParser(description="LSS Standalone PCA Fitting Tool")
    parser.add_argument("-i", "--input", help="Directory of clean audio files for PCA estimation")
    parser.add_argument("-o", "--output_dir", help="Directory to save the computed PCA .pt file")
    parser.add_argument("-d", "--device", type=int, help="Physical GPU device ID (overrides config file)")
    parser.add_argument("--config", default="config/config_LSS.yaml", help="Path to configuration file")
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process for fitting")
    parser.add_argument("--fit_pca", action="store_true", default=None, help="Estimate PCA on selected input set")
    parser.add_argument("--eval_mode", action="store_true", default=False, help="Evaluation mode")

    args = parser.parse_args()

    # Read config once
    config = read_yaml(args.config)

    # Determine and set device
    target_device = args.device if args.device is not None else config.get('device', -1)
    set_gpu(target_device)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Extract mode flags from configuration
    load_pca = config['modes']['load_pca']
    fit_pca = config['modes']['fit_pca']
    save_pca = config['modes']['save_pca']
    eval_mode = config['modes']['eval_mode']
    pca_load_path = config['paths']['pca_load_dir']

    input_target = args.input or config['paths']['pca_estim_audio_dir']
    files = discover_audio_paths(input_target)

    # Initialize model
    enc = EncodecLatent_NEW(device=device)
    lsp = LatentSpinPCA(
        key_hex="00" * 16, theta=0, P_payload=1, P_pilots=0, M=32, L=8, mixers=0, repetition=1, device=device
    )

    max_fit = args.max_files or config['script'].get('max_files', len(files))
    pca_is_loaded = False

    # --- A. Attempt Loading ---
    # We load ONLY if:
    # 1. load_pca is True
    # 2. The file actually exists
    if load_pca and not fit_pca:
        if os.path.exists(pca_load_path) and os.path.isfile(pca_load_path):
            try:
                # Use helper to load stats
                W, Winv, mu, lam = load_pca_stats(pca_load_path, device)
                lsp.W, lsp.Winv, lsp.mu, lsp.lam = W, Winv, mu, lam
                pca_is_loaded = True
                print("[PCA] Loaded successfully. Skipping fitting.")
            except Exception as e:
                print(f"[WARN] Failed to load PCA state from {pca_load_path}: {e}")
                print("[PCA] Will fall back to fresh fitting.")
        elif fit_pca:
            print(f"[PCA] load_pca=True, but file not found or invalid: '{pca_load_path}'")
            print("[PCA] Will fall back to fresh fitting.")

    # --- B. Fit (Fallback / Default) ---
    if not pca_is_loaded and fit_pca:
        if not files:
            print(f"[FATAL] No valid audio assets located at: {input_target}")
            return

        if len(files) > max_fit:
            random.seed(config['split']['seed'])
            files = random.sample(files, max_fit)

        print(f"\n[PHASE 1] Fitting PCA on Train Set (limited to {max_fit} samples)...")
        dataset = AudioDataset(file_paths=files, target_sr=enc.enc_samplerate, duration=10.0, mode='test')

        train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=custom_collate_asv,
            num_workers=config['batching']['num_workers']
        )

        collected_latents = []

        with torch.no_grad():
            for batch_wavs, _ in tqdm(train_loader, desc="PCA Fit:", total=max_fit):
                if len(collected_latents) >= max_fit: break
                if batch_wavs is None: continue

                for wav in batch_wavs:
                    F = enc.encode_train_pca(wav.to(device)).squeeze(0).cpu()
                    collected_latents.append(F)

        print(f"[PCA] Concatenating {len(collected_latents)} files for SVD...")
        W, Winv, mu, lam = pca_fit(collected_latents)

        # Assign to model
        lsp.W, lsp.Winv, lsp.mu, lsp.lam = W.to(device), Winv.to(device), mu.to(device), lam.to(device)
        print("[PCA] Fit Complete.")

        # --- Save and Exit (If fitting) ---
        if fit_pca or save_pca:
            if args.output_dir:
                config['paths']['pca_save_dir'] = args.output_dir

            # Pass the length of collected latents to satisfy your utils.py save_pca_stats signature
            pca_save_path = save_pca_stats(lsp, len(collected_latents), config)

            if fit_pca and not eval_mode:
                print("-" * 60)
                print(f"[DONE] PCA fitted and saved to: {pca_save_path}")
                print("Skipping evaluation phase. Exiting script.")
                print("-" * 60)
                return


if __name__ == "__main__":
    main()