import os
import argparse
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import *
from src.encodec_wrapper import EncodecLatent_NEW
from src.lss_pca import LatentSpinPCA
from src.audio_processing import AudioDataset, custom_collate_asv



def main():


    parser = argparse.ArgumentParser(description="LSS Embedding")
    parser.add_argument("-i", "--input", help="Path to single audio file or folder directory")
    parser.add_argument("-o", "--output_dir",  help="Directory path to save watermarked wav audio")
    parser.add_argument("-d", "--device", type=int, help="Physical GPU device ID (overrides config file)")
    parser.add_argument("--config", default="config/config_LSS.yaml", help="Path to configuration file")
    parser.add_argument("--pca_load_dir", help="Directory path for estimated PCA basis")
    parser.add_argument("--key", help="Override secret key sequence string")
    parser.add_argument("--theta", type=float, help="Override rotation angle magnitude (radians)")
    parser.add_argument("--bits", help="Override payload bit string")

    args = parser.parse_args()

    config = read_yaml(args.config)
    # Determine and set device
    target_device = args.device if args.device is not None else config.get('device', -1)
    set_gpu(target_device)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configuration fallback priority: CLI arg -> Config key -> Error
    input_target = args.input or config['paths']['embed_audio_dir']
    output_destination = args.output_dir or config['paths']['audio_wm_out_dir']

    if not input_target or not output_destination:
        print("[FATAL] Missing required input target path or output destination directory.")
        return

    # Updated to match your configuration key change
    pca_path = args.pca_load_dir or config['paths']['pca_load_dir']
    secret_key = args.key or config['run_lss']['key']
    theta = args.theta or config['pca_lss']['theta']
    bit_str = args.bits or config['script']['bits']
    bits = [1 if c == '1' else 0 for c in bit_str if c in '01']

    if not secret_key or not bits:
        print("[FATAL] A valid secret key and payload bit sequence must be specified.")
        return

    files = discover_audio_paths(input_target)
    if not files:
        print(f"[FATAL] No valid audio tracks located at target path: {input_target}")
        return

    print(f"[INFO] Initializing EnCodec backend and loading PCA basis stats...")

    enc = EncodecLatent_NEW(device=device)

    lsp = LatentSpinPCA(
        key_hex=secret_key,
        theta=theta,
        P_payload=config['pca_lss']['payload_planes'],
        P_pilots=config['pca_lss']['pilots'],
        M=config['pca_lss']['m_frames'],
        L=config['pca_lss']['L'],
        mixers=config['pca_lss']['householder_mixers'],
        repetition=config['pca_lss']['repetition_default'],
        device=device
    )

    try:
        W, Winv, mu, lam = load_pca_stats(pca_path, device)
        lsp.W, lsp.Winv, lsp.mu, lsp.lam = W, Winv, mu, lam
    except Exception as e:
        print(f"[WARN] Invalid or missing PCA basis at '{pca_path}'. Error: {e}")
        print("[FATAL] Embedding requires a valid PCA basis. Exiting script.")
        return

    dataset = AudioDataset(file_paths=files, target_sr=enc.enc_samplerate, duration=10.0, mode='test')
    loader = DataLoader(
        dataset,
        batch_size=config['batching']['test_batch_size'],
        shuffle=False,
        collate_fn=custom_collate_asv,
        num_workers=config['batching']['num_workers']
    )

    os.makedirs(output_destination, exist_ok=True)
    print(f"[INFO] Launching batch embedding loop across {len(files)} items.")

    with torch.no_grad():
        for batch_wavs, batch_ids in tqdm(loader, desc="Embedding Batch Progress"):
            if batch_wavs is None:
                continue

            batch_wavs = batch_wavs.to(device)
            try:
                F_orig = enc.encode_test_batch(batch_wavs)
                F_wm = lsp.embed_features_batch(F_orig, batch_ids, bits)
                wavs_wm = enc.decode_from_features(F_wm)

                for idx, fid in enumerate(batch_ids):
                    out_path = os.path.join(output_destination,f"{fid}.wav")
                    torchaudio.save(str(out_path), wavs_wm[idx].cpu(), enc.enc_samplerate)
            except Exception as e:
                print(f"[ERR] Continuous processing loop failure: {e}")

    print(f"[SUCCESS] Clean watermarked audio successfully exported to: {output_destination}")


if __name__ == "__main__":

    main()