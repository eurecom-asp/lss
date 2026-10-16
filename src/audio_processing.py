import os
import shutil
import tempfile
import subprocess
import scipy.signal
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

from src.utils import read_yaml

# Resolve config path relative to src/ folder location
src_folder = Path(__file__).parent
config_folder = src_folder.parent
config_path = str(config_folder / "config" / "config_LSS.yaml")
config = read_yaml(config_path)


# =============================================================================
#                        GENERIC BATCH AUDIO DATASET
# =============================================================================

class AudioDataset(Dataset):
    """
    A generic audio dataset that loads, resamples, and pads/crops waveforms
    to a fixed duration for uniform batch processing.
    """

    def __init__(self,
                 file_paths: List[Path],
                 target_sr: int = config['encodec']['sample_rate'],
                 duration: float = 10.0,
                 mode: str = 'test'):
        """
        Args:
            file_paths: List of Path objects to the audio assets.
            target_sr: Codec target sample rate (e.g., 24000).
            duration: Target duration in seconds.
            mode: 'train' (raw length) or 'test' (fixed length via padding/cropping).
        """
        # Convert strings to Path objects
        self.file_paths = [Path(p) for p in file_paths]

        self.target_sr = target_sr
        self.duration = duration
        self.target_samples = int(target_sr * duration)
        self.mode = mode

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            fpath = self.file_paths[idx]
            file_id = fpath.stem

            if not fpath.exists():
                print(f"[WARN] File not found: {fpath}")
                return None, file_id

            # 1. Load Audio
            wav, sr = torchaudio.load(str(fpath))

            # 2. Resample dynamically to the codec sample rate
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)

            # Ensure single channel mono [1, T]
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            current_samples = wav.shape[-1]

            # 3. Dynamic Crop / Circular Pad Logic (from paper specification)
            if self.mode != 'train' and self.duration is not None and self.duration > 0:
                # Too Short -> Circular Padding
                if current_samples < self.target_samples:
                    n_repeats = (self.target_samples // current_samples) + 1
                    wav = wav.repeat(1, n_repeats)
                    wav = wav[:, :self.target_samples]
                # Too Long -> Truncate
                elif current_samples > self.target_samples:
                    wav = wav[:, :self.target_samples]

            return wav, file_id

        except Exception as e:
            print(f"[ERR] Failed loading {self.file_paths[idx].name}: {e}")
            return None, "ERROR"


def custom_collate_asv(batch):
    """
    Filters loading errors and stacks tensors for efficient batching.
    """
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None

    wavs, ids = zip(*batch)
    return torch.stack(wavs), list(ids)


# =============================================================================
#                       FILTERING MANIPULATIONS
# =============================================================================

def lowpass_zero_phase(wav: torch.Tensor, cutoff_hz: float, sr: int, wm_base_dir: str = None, file_name: str = None):
    original_shape = wav.shape
    device = wav.device

    if wav.dim() == 3:
        wav_core = wav.squeeze(0).squeeze(0)
    elif wav.dim() == 2:
        wav_core = wav.squeeze(0)
    else:
        wav_core = wav

    sos = scipy.signal.butter(N=6, Wn=cutoff_hz, btype='low', fs=sr, output='sos')
    wav_np = wav_core.detach().cpu().numpy()
    wav_filt_np = scipy.signal.sosfiltfilt(sos, wav_np)

    if wav_filt_np.strides[0] < 0:
        wav_filt_np = wav_filt_np.copy()

    wav_filt_core = torch.from_numpy(wav_filt_np).float().to(device)

    if len(original_shape) == 3:
        wav_filt = wav_filt_core.unsqueeze(0).unsqueeze(0)
    elif len(original_shape) == 2:
        wav_filt = wav_filt_core.unsqueeze(0)
    else:
        wav_filt = wav_filt_core

    if wm_base_dir is not None and file_name is not None:
        out_dir = os.path.join(wm_base_dir, f"lowpass_{int(cutoff_hz)}Hz")
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(os.path.join(out_dir, f"{file_name}.wav"), wav_filt.squeeze(0).cpu(), sample_rate=sr)

    return wav_filt


def highpass_zero_phase(wav: torch.Tensor, cutoff_hz: float, sr: int, wm_base_dir: str = None, file_name: str = None) -> torch.Tensor:
    original_shape = wav.shape
    device = wav.device

    if wav.dim() == 3:
        wav = wav.squeeze(0).squeeze(0)
    elif wav.dim() == 2:
        wav = wav.squeeze(0)

    sos = scipy.signal.butter(N=6, Wn=cutoff_hz, btype='high', fs=sr, output='sos')
    wav_np = wav.detach().cpu().numpy()
    wav_filt_np = scipy.signal.sosfiltfilt(sos, wav_np)

    if any(s < 0 for s in wav_filt_np.strides):
        wav_filt_np = wav_filt_np.copy()

    wav_filt = torch.from_numpy(wav_filt_np).float().to(device)

    if len(original_shape) == 3:
        wav_filt = wav_filt.unsqueeze(0).unsqueeze(0)
    else:
        wav_filt = wav_filt.unsqueeze(0)

    if wm_base_dir is not None and file_name is not None:
        out_dir = os.path.join(wm_base_dir, f"highpass_{int(cutoff_hz)}Hz")
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(os.path.join(out_dir, f"{file_name}.wav"), wav_filt.squeeze(0).cpu(), sample_rate=sr)

    return wav_filt


def bandpass_zero_phase(wav: torch.Tensor, cutoff_low_hz: float, cutoff_high_hz: float, sr: int, wm_base_dir: str = None, file_name: str = None) -> torch.Tensor:
    original_shape = wav.shape
    device = wav.device

    if wav.dim() == 3:
        wav_core = wav.squeeze(0).squeeze(0)
    elif wav.dim() == 2:
        wav_core = wav.squeeze(0)
    else:
        wav_core = wav

    sos = scipy.signal.butter(N=6, Wn=[cutoff_low_hz, cutoff_high_hz], btype='band', fs=sr, output='sos')
    wav_np = wav_core.detach().cpu().numpy()
    wav_filt_np = scipy.signal.sosfiltfilt(sos, wav_np)

    if any(s < 0 for s in wav_filt_np.strides):
        wav_filt_np = wav_filt_np.copy()

    wav_filt = torch.from_numpy(wav_filt_np).float().to(device)

    if len(original_shape) == 3:
        wav_filt = wav_filt.unsqueeze(0).unsqueeze(0)
    elif len(original_shape) == 2:
        wav_filt = wav_filt.unsqueeze(0)

    if wm_base_dir is not None and file_name is not None:
        out_dir = os.path.join(wm_base_dir, f"bandpass_{int(cutoff_low_hz)}_{int(cutoff_high_hz)}Hz")
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(os.path.join(out_dir, f"{file_name}.wav"), wav_filt.squeeze(0).cpu(), sample_rate=sr)

    return wav_filt


# =============================================================================
#                       MP3 & RESAMPLING ATTACKS
# =============================================================================

# =========================================================
#                       MP3 COMPRESSION
# =========================================================

def save_mp3(path, wav_tensor, sr):
    torchaudio.save(path, wav_tensor.unsqueeze(0).cpu(), sr, format="mp3")

def apply_mp3(
    wav: torch.Tensor,
    sr: int = config['encodec']['sample_rate'],
    bitrate_kbps: int = config['manipulation']['mp3_bitrate'],
    ffmpeg_path: Optional[str] = None,
    wm_base_dir: str = None,
    file_name: str = None,
) -> torch.Tensor:
    """
    Compress & decompress audio using MP3 via ffmpeg.
    Operates in-memory; optionally saves to disk if wm_base_dir is provided.
    Automatically matches the output tensor dimensions to the input tensor.
    """
    device = wav.device
    original_dim = wav.dim()

    if wav.dim() == 3:
        wav = wav.squeeze(0)

    if wav.dim() != 2:
        raise ValueError(f"Expected waveform shape [1,T] or [1,1,T], got {wav.shape}")

    if ffmpeg_path is None:
        ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None or not shutil.which(ffmpeg_path):
        raise RuntimeError("ffmpeg not found. Install it in your conda env or provide its path explicitly.")

    wav_cpu = wav.detach().cpu().float()
    mp3_save_path = None

    # Prepare output dir if requested
    if wm_base_dir is not None and file_name is not None:
        save_mp3_dir = os.path.join(wm_base_dir, f'mp3_{bitrate_kbps}kbps')
        os.makedirs(save_mp3_dir, exist_ok=True)
        mp3_save_path = os.path.join(save_mp3_dir, f"{file_name}.mp3")

    with tempfile.TemporaryDirectory(prefix=f"mp3_{bitrate_kbps}_") as tmp:
        wav_in = os.path.join(tmp, "in.wav")
        mp3_out = os.path.join(tmp, "out.mp3")
        wav_out = os.path.join(tmp, "out.wav")

        torchaudio.save(wav_in, wav_cpu, sr)

        subprocess.run(
            [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
             "-i", wav_in, "-b:a", f"{bitrate_kbps}k", mp3_out],
            check=True
        )

        # Save MP3 permanently if requested
        if mp3_save_path:
            shutil.copy(mp3_out, mp3_save_path)

        subprocess.run(
            [ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
             "-i", mp3_out, wav_out],
            check=True
        )

        decoded, sr_dec = torchaudio.load(wav_out)

        if sr_dec != sr:
            decoded = torchaudio.functional.resample(decoded, sr_dec, sr)

        if original_dim == 3 and decoded.dim() == 2:
            decoded = decoded.unsqueeze(0)
        elif original_dim == 2 and decoded.dim() == 3:
            decoded = decoded.squeeze(0)

        return decoded.to(device)


def apply_resampling(wav: torch.Tensor, target_sr: int, orig_sr: int, wm_base_dir: str = None, file_name: str = None) -> torch.Tensor:
    device = wav.device
    original_length = wav.shape[-1]

    wav_down = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=target_sr)

    if wm_base_dir is not None and file_name is not None:
        out_dir = os.path.join(wm_base_dir, f"resample_{target_sr}")
        os.makedirs(out_dir, exist_ok=True)
        torchaudio.save(os.path.join(out_dir, f"{file_name}.wav"), wav_down.cpu(), sample_rate=target_sr)

    wav_up = torchaudio.functional.resample(wav_down, orig_freq=target_sr, new_freq=orig_sr)
    current_len = wav_up.shape[-1]

    if current_len > original_length:
        wav_up = wav_up[..., :original_length]
    elif current_len < original_length:
        diff = original_length - current_len
        if wav_up.dim() == 1:
            wav_up = wav_up.unsqueeze(0)
        wav_up = torch.nn.functional.pad(wav_up, (0, diff), mode='circular')

    return wav_up.to(device)


# =============================================================================
#                              NOISE GENERATORS
# =============================================================================

def generate_pink_noise(shape: torch.Size, device: torch.device) -> torch.Tensor:
    noise = torch.randn(shape, device=device)
    fft = torch.fft.rfft(noise, dim=-1)
    freqs = torch.arange(fft.shape[-1], device=device) + 1.0
    scale = 1.0 / torch.sqrt(freqs)
    fft_pink = fft * scale
    return torch.fft.irfft(fft_pink, n=shape[-1], dim=-1)


def mix_noise_snr(clean_audio: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    p_signal = torch.mean(clean_audio ** 2)
    p_noise_current = torch.mean(noise ** 2)

    if p_signal == 0:
        return clean_audio

    p_noise_target = p_signal / (10 ** (snr_db / 10.0))
    scaling_factor = torch.sqrt(p_noise_target / (p_noise_current + 1e-9))
    return torch.clamp(clean_audio + (noise * scaling_factor), -1.0, 1.0)


# =============================================================================
#                            BATCH DISPATCH ENGINE
# =============================================================================

def apply_manipulation_batch(wav_batch, config, out_dir, batch_ids, target_sr):
    manip_name = config['manipulation']['name']
    if manip_name == 'clean':
        return wav_batch

    cutoff_low = float(config['manipulation']['cutoff_low'])
    cutoff_high = float(config['manipulation']['cutoff_high'])
    mp3_bitrate = int(config['manipulation']['mp3_bitrate'])
    cutoff_bp_low = float(config['manipulation']['cutoff_bp_low'])
    cutoff_bp_high = float(config['manipulation']['cutoff_bp_high'])
    target_resample = int(config['manipulation']['target_resample'])
    snr_db = float(config['manipulation']['snr_db'])

    output_batch = []
    for i, wav in enumerate(wav_batch):
        fid = batch_ids[i]
        try:
            if manip_name == 'lowpass_hz':
                wav_manip = lowpass_zero_phase(wav, cutoff_hz=cutoff_low, sr=target_sr, wm_base_dir=out_dir, file_name=fid)
            elif manip_name == 'highpass_hz':
                wav_manip = highpass_zero_phase(wav, cutoff_hz=cutoff_high, sr=target_sr, wm_base_dir=out_dir, file_name=fid)
            elif manip_name == 'bandpass_hz':
                wav_manip = bandpass_zero_phase(wav, cutoff_low_hz=cutoff_bp_low, cutoff_high_hz=cutoff_bp_high, sr=target_sr, wm_base_dir=out_dir, file_name=fid)
            elif manip_name == 'mp3_kbps':
                wav_manip = apply_mp3(wav, bitrate_kbps=mp3_bitrate, sr=target_sr, wm_base_dir=out_dir, file_name=fid)
            elif manip_name == 'resampling':
                wav_manip = apply_resampling(wav, target_sr=target_resample, orig_sr=target_sr, wm_base_dir=out_dir, file_name=fid)
            elif manip_name == 'white_noise':
                wav_manip = mix_noise_snr(wav, torch.randn_like(wav), snr_db)
            elif manip_name == 'pink_noise':
                wav_manip = mix_noise_snr(wav, generate_pink_noise(wav.shape, wav.device), snr_db)
            else:
                wav_manip = wav
        except Exception as e:
            print(f"[WARN] Manipulation {manip_name} failed for {fid}: {e}")
            wav_manip = wav

        output_batch.append(wav_manip)

    return torch.stack(output_batch).to(wav_batch.device)