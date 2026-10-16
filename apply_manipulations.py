import os
import argparse
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import *
from src.audio_processing import AudioDataset, custom_collate_asv, apply_manipulation_batch


def main():



    parser = argparse.ArgumentParser(description="LSS Standalone DSP Manipulation Tool")
    parser.add_argument("-i", "--input", required=True, help="Path to watermarked folder or single audio file")
    parser.add_argument("-o", "--output_dir", required=True,help="Directory path to save manipulated wav assets")
    parser.add_argument("-d", "--device", type=int, help="Physical GPU device ID (overrides config file)")
    parser.add_argument("--config", default="config/config_LSS.yaml", help="Path to configuration layout file")
    parser.add_argument("--attack",
                        choices=['lowpass_hz', 'highpass_hz', 'bandpass_hz', 'mp3_kbps', 'resampling', 'white_noise',
                                 'pink_noise'],
                        help="Override the type of signal distortion technique applied")
    parser.add_argument("--param", type=float, help="Override the core channel value (e.g. cutoff Hz, bitrate, SNR dB)")
    parser.add_argument("--save_manipulated_wav", action="store_true", default=None,
                        help="Explicitly force saving the mutated files to disk")
    args = parser.parse_args()

    config = read_yaml(args.config)
    set_gpu(config['device'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine saving status cascade: argparse flag -> yaml setting -> default False
    if args.save_manipulated_wav is not None:
        save_manipulated_wav = args.save_manipulated_wav
    else:
        save_manipulated_wav = config['run_lss'].get('save_manipulated_wav', False)

    # Handle dynamic configuration updates
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

    files = discover_audio_paths(args.input)
    if not files:
        print(f"[FATAL] No valid audio assets located at: {args.input}")
        return

    # Use the same exact target sample rate as the encoder to process the signals correctly
    target_sr = config['encodec']['sample_rate']
    dataset = AudioDataset(file_paths=files, target_sr=target_sr, duration=10.0, mode='test')
    loader = DataLoader(
        dataset,
        batch_size=config['batching']['test_batch_size'],
        shuffle=False,
        collate_fn=custom_collate_asv,
        num_workers=config['batching']['num_workers']
    )

    print(f"[INFO] Running channel distortion attack: '{manip_name}' across {len(files)} tracks.")
    print(f"[INFO] Export audio saving configuration condition: {save_manipulated_wav}")

    # Generate dynamic subfolder name based on the attack type and parameters
    if manip_name == 'lowpass_hz':
        m_dir = f"lowpass_{int(config['manipulation']['cutoff_low'])}Hz"
    elif manip_name == 'highpass_hz':
        m_dir = f"highpass_{int(config['manipulation']['cutoff_high'])}Hz"
    elif manip_name == 'bandpass_hz':
        m_dir = f"bandpass_{int(config['manipulation']['cutoff_bp_low'])}_{int(config['manipulation']['cutoff_bp_high'])}Hz"
    elif manip_name == 'mp3_kbps':
        m_dir = f"mp3_{config['manipulation']['mp3_bitrate']}kbps"
    elif manip_name == 'resampling':
        m_dir = f"resample_{target_sr}_to_{int(config['manipulation']['target_resample'])}Hz"
    elif manip_name in ['white_noise', 'pink_noise']:
        m_dir = f"{manip_name}_{config['manipulation']['snr_db']}dB"
    else:
        m_dir = manip_name

    final_output_dir = os.path.join(args.output_dir, m_dir)

    if save_manipulated_wav:
        os.makedirs(final_output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_wavs, batch_ids in tqdm(loader, desc="DSP Attack Progress"):
            if batch_wavs is None:
                continue

            batch_wavs = batch_wavs.to(device)
            try:
                # Apply the native zero-phase filtering/noise algorithms from audio_processing
                wavs_manip = apply_manipulation_batch(batch_wavs, config, out_dir=None, batch_ids=batch_ids, target_sr=target_sr)

                if save_manipulated_wav:
                    for idx, fid in enumerate(batch_ids):
                        out_path = os.path.join(final_output_dir, f"{fid}.wav")
                        torchaudio.save(out_path, wavs_manip[idx].cpu(), target_sr)
            except Exception as e:
                print(f"[ERR] Audio manipulation failure on processing loop: {e}")

    if save_manipulated_wav:
        print(f"[SUCCESS] Distorted waveforms successfully exported to: {final_output_dir}")
    else:
        print("[SUCCESS] Processing batch complete. Audio files were not saved to disk.")


if __name__ == "__main__":

    main()