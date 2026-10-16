import os
from typing import Tuple, Optional
from copy import deepcopy
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import read_yaml
import torch
import torchaudio
from transformers import EncodecModel

src_folder = Path(__file__).parent
config_folder = src_folder.parent
config_path = str(config_folder / "config" / "config_LSS.yaml")
config = read_yaml(config_path)


# =============================================================================
#                              EnCodec wrapper
# =============================================================================



class QuantizerPassThrough(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def encode(self, x, bandwidth):
        # 1. Fix the Crash: Unwrap the tuple from the Encoder
        if isinstance(x, tuple):
            x = x[0]
        elif hasattr(x, "last_hidden_state"):
            x = x.last_hidden_state

        # 2. Return the raw tensor.
        # The library will transpose this to [128, B, T]. We will fix that later.
        return x


class EncodecLatent_NEW:
    """
    Wrapper for Encodec (HuggingFace version).
    Methods:
      - encode_train_pca(wav) -> [1, 128, T] (Strictly for B=1)
      - encode_test_batch(wav) -> [B, 128, T] (For B>=1)
      - decode_from_features(F) -> [B, 1, T]
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        self.src_samplerate = config['input_audio']['sample_rate']
        self.enc_samplerate = config['encodec']['sample_rate']
        self.target_bw_kbps = config['encodec']['target_bandwidth_kbps']

        # Load Model ONCE
        self.normal_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.no_vq_model = EncodecModel.from_pretrained("facebook/encodec_24khz")

        # Apply the Tuple-Fix Quantizer
        self.absolutely_useless_quantizer = QuantizerPassThrough()
        self.original_quantizer = deepcopy(self.no_vq_model.quantizer)
        self.no_vq_model.quantizer = self.absolutely_useless_quantizer

        self.normal_model.to(self.device).eval()
        self.no_vq_model.to(self.device).eval()
        self.original_quantizer.to(self.device)

        # High-quality resampler (16k -> 24k)
        self.resample_16_to_24 = torchaudio.transforms.Resample(
            orig_freq=self.src_samplerate, new_freq=self.enc_samplerate,
            lowpass_filter_width=64, rolloff=0.945,
            resampling_method="sinc_interp_kaiser", beta=14.7696564594
        ).to(self.device)

    @torch.no_grad()
    def load_audio(self, path: str) -> torch.Tensor:
        """Helper to load single audio file [1, T]."""
        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = wav.to(self.device)

        if sr != self.enc_samplerate:
            if sr == self.src_samplerate and self.enc_samplerate == 24000:
                wav = self.resample_16_to_24(wav)
            else:
                wav = torchaudio.functional.resample(wav, sr, self.enc_samplerate)
        return wav

    @torch.no_grad()
    def encode_train_pca(self, wav_1_sample: torch.Tensor) -> torch.Tensor:
        """
        PHASE 1 (PCA Fit): STRICTLY for Batch Size = 1.
        Output: [1, 128, T]
        """
        if wav_1_sample.dim() == 2:
            wav_1_sample = wav_1_sample.unsqueeze(1)

        B = wav_1_sample.shape[0]
        assert B == 1, f"encode_train_pca expects B=1, got {B}."

        encodec_output = self.no_vq_model.encode(wav_1_sample, bandwidth=self.target_bw_kbps)
        feats = encodec_output.audio_codes

        if feats.dim() == 4 and feats.shape[1] == 1:
            feats = feats.squeeze(1)
        elif feats.dim() == 4 and feats.shape[2] == 1:
            feats = feats.squeeze(2)

        return feats.contiguous()


    @torch.no_grad()
    def encode_test_batch(self, wav_batch: torch.Tensor) -> torch.Tensor:
        """
        Output: [B, 128, T]
        """
        input_batch_size = wav_batch.shape[0]

        # 1. Standardize Input
        if wav_batch.dim() == 2:
            wav_batch = wav_batch.unsqueeze(1)

        # 2. Run Model (This will now return [128, B, T] because of the library's internal transpose)
        encodec_outputs = self.no_vq_model.encode(wav_batch, bandwidth=self.target_bw_kbps)
        feats = encodec_outputs.audio_codes     #Tensor: [1, 128, 32, 750]

        # 3. Fix Dimensions
        # Step A: Remove the '1' at dim 0 -> [128, 32, 750]
        if feats.dim() == 4 and feats.shape[0] == 1:
            feats = feats.squeeze(0)

        # Step B: Swap Channels (128) and Batch (32) -> [32, 128, 750]
        if feats.shape[1] == input_batch_size:
            feats = feats.transpose(0, 1)

        return feats.contiguous()

    @torch.no_grad()
    def decode_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Decodes continuous features back to audio for BATCHES.
        Output: [B, 1, T_audio]
        """
        input_batch_size = feats.shape[0]

        if feats.dim() == 2:
            feats = feats.unsqueeze(0)

        feats = feats.to(self.device)

        # 1. Apply Original Quantization
        # RETURNS A TUPLE: (quantized, codes, loss)
        q_codes = self.original_quantizer.encode(feats, bandwidth=self.target_bw_kbps)

        # 3. Format for Decoder
        # Target: [Chunks=1, Batch, Codebooks, Time]
        codes = q_codes.transpose(0, 1).unsqueeze(0)  # [1, B, 8, T_enc]

        # 4. Decode
        audio_scales = [None]
        audio_values = self.normal_model.decode(codes, audio_scales)

        if isinstance(audio_values, tuple):
            y = audio_values[0]
        else:
            y = audio_values.audio_values

        return y

