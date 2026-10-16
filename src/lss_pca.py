import os
import math
import struct
import hmac
import hashlib
import random
import numpy as np
import argparse
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from copy import deepcopy

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import *
from audio_processing import *
from encodec_wrapper import *

import torch
import torchaudio
import inspect

# Encodec (NEW!)
# from encodec import EncodecModel
from transformers import EncodecModel

src_folder = Path(__file__).parent
config_folder = src_folder.parent

config_path = str(config_folder / "config" / "config_LSS.yaml")

config = read_yaml(config_path)

# =============================================================================
#                              LSS over PCA features
# =============================================================================

class LatentSpinPCA:
    def __init__(self,
                 key_hex: str,
                 theta: float,
                 P_payload: int,
                 P_pilots: int,
                 M: int,
                 L: int,
                 mixers: int,
                 repetition: int,
                 device: Optional[str] = None):

        # KEY instantiation
        self.key = (bytes.fromhex(key_hex)
                    if all(c in '0123456789abcdef' for c in key_hex.lower())
                    else key_hex.encode())

        assert len(self.key) >= 16, "Use at least a 128-bit key."

        # EMBEDDING params
        self.theta = float(theta)
        self.P_req = int(P_payload)
        self.P0 = int(P_pilots)
        self.M = int(M)
        self.mixers = int(mixers)
        self.rep = int(repetition)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.L = int(L)  # number of temporal chips

        # PCA params
        self.W = None  # [n×n], Z = W(F-μ)
        self.Winv = None  # [n×n], F ≈ Winv Z + μ
        self.mu = None  # [n×1]
        self.lam = None  # [n], eigenvalues

    @torch.no_grad()
    def fit_pca(self, enc: EncodecLatent_NEW, items: List[Tuple[str, str, str]], max_files: int = 200):
        latents = []
        for k, (_, fid, path) in enumerate(items[:max_files]):
            wav = enc.load_audio(path)
            F = enc.encode_to_features(wav)  # [n, T']
            F = F.squeeze(0)
            latents.append(F.to('cpu'))
        W, Winv, mu, lam = pca_fit(latents)
        self.W, self.Winv, self.mu = W.to(self.device), Winv.to(self.device), mu.to(self.device)
        self.lam = lam.to(self.device)

    def load_pca_from(self, other: "LatentSpinPCA"):
        """Copy PCA params from another instance (so we reuse the same PCA basis for wrong-key eval)."""
        self.W = other.W
        self.Winv = other.Winv
        self.mu = other.mu
        self.lam = other.lam

    def _check_pca(self):
        assert self.W is not None and self.Winv is not None and self.mu is not None and self.lam is not None, \
            "PCA not set; call fit_pca(...) or load_pca_from(...)."

    @staticmethod
    def derive_nonce(key: bytes, file_id: str) -> bytes:
        return hmac.new(key, f"nonce::{file_id}".encode(), hashlib.sha256).digest()[:12]

    # --------- Anisotropic keyed plane selection with canonical orientation ---------

    def _anisotropic_pairs(self, nonce: bytes, chunk_idx: int, n: int, P_eff: int):
        """
        Pick (P0 + P_eff) disjoint planes among high-variance dims, with canonical orientation:
        the first index in each pair has the larger eigenvalue λ. We pair a keyed-shuffled
        top-half with a keyed-shuffled next-half from the top-K dims to increase anisotropy.
        """
        total_pairs = self.P0 + P_eff
        if total_pairs <= 0:
            return []

        K = int(min(n, 2 * total_pairs))  # number of dims to draw from
        idx_sorted = torch.argsort(self.lam, descending=True)[:K]  # top-K by energy

        H = min(total_pairs, K // 2)
        top_half = idx_sorted[:H]
        next_half = idx_sorted[H:2 * H]

        if next_half.numel() < H:  # shrink if needed
            H = next_half.numel()
            top_half = top_half[:H]
            next_half = next_half[:H]

        noise_top = prf_gaussian(self.key, b'IDX_TOP' + nonce + struct.pack('>I', chunk_idx), (H,)).to(self.device)
        noise_next = prf_gaussian(self.key, b'IDX_NEXT' + nonce + struct.pack('>I', chunk_idx), (H,)).to(self.device)

        top_shuf = top_half[torch.argsort(noise_top)]
        next_shuf = next_half[torch.argsort(noise_next)]

        pairs = []
        for a, b in zip(top_shuf.tolist(), next_shuf.tolist()):
            # Canonical orientation: first index has larger lambda
            i, j = (a, b) if self.lam[a] >= self.lam[b] else (b, a)
            pairs.append((int(i), int(j)))
            if len(pairs) >= total_pairs:
                break
        return pairs[:total_pairs]

    # --------- Embed: rotate PCA features (returns F_wm, F_orig, meta) ---------

    @torch.no_grad()
    def embed_features_batch(self, F_batch: torch.Tensor, file_ids: List[str], bits: List[int]) -> torch.Tensor:

        self._check_pca()

        F_batch = F_batch.to(self.device)
        B, n, T = F_batch.shape
        assert len(file_ids) == B

        bits_coded = [bit for bit in bits for _ in range(self.rep)]

        P_cap = max(1, n // 2 - self.P0)
        P_eff = max(1, min(self.P_req, P_cap))

        # PCA projection
        Z_batch = torch.matmul(self.W, F_batch - self.mu)

        for b in range(B):

            nonce = self.derive_nonce(self.key, file_ids[b])
            Z_sample = Z_batch[b]

            bit_idx = 0

            for c, (s, e) in enumerate(chunk_ranges(T, self.M)):

                Qm = householder_chain_Q(
                    self.key, nonce, c, n,
                    m=self.mixers,
                    device=self.device
                )

                Zhat = torch.matmul(Qm, Z_sample[:, s:e])

                pairs = self._anisotropic_pairs(nonce, c, n, P_eff)
                pilots = pairs[:self.P0]
                payload_pairs = pairs[self.P0:] if self.P0 > 0 else pairs

                for (i, j) in pilots:
                    givens_apply_(Zhat, i, j, +self.theta)

                # chip structure
                Lc = e - s
                sub = max(1, Lc // self.L)

                num_planes = len(payload_pairs)

                # ---- generate all chips for the chunk at once ----
                chips_all = prf_gaussian(
                    self.key,
                    b'CHIP' + nonce + struct.pack(">I", c),
                    (num_planes, self.L),
                    device=self.device
                ).sign()

                for p, (i, j) in enumerate(payload_pairs):

                    chips = chips_all[p]

                    bit = bits_coded[bit_idx % len(bits_coded)]
                    beta = 1.0 if bit == 1 else -1.0

                    for ell in range(self.L):

                        ss = ell * sub
                        ee = (ell + 1) * sub if ell < self.L - 1 else Lc

                        if ss >= ee:
                            continue

                        phi = beta * chips[ell] * self.theta

                        givens_apply_(Zhat[:, ss:ee], i, j, phi)

                    bit_idx += 1

                Z_sample[:, s:e] = torch.matmul(Qm.T, Zhat)

        F_wm_batch = torch.matmul(self.Winv, Z_batch)
        F_wm_batch += self.mu

        return F_wm_batch


    @torch.no_grad()
    def decode_from_features(self, enc: EncodecLatent_NEW, F_wm: torch.Tensor) -> torch.Tensor:
        return enc.decode_from_features(F_wm)

    # --------- Presence score in PCA space (cross-covariance GLRT) ---------


    @torch.no_grad()
    def presence_score_from_features(self, F: torch.Tensor, file_id: str, bits: List[int]) -> float:

        self._check_pca()

        if F.dim() == 3 and F.shape[0] == 1:
            F = F.squeeze(0)

        F = F.to(self.device)

        n, T = F.shape

        P_cap = max(1, n // 2 - self.P0)
        P_eff = max(1, min(self.P_req, P_cap))

        bits_coded = [bit for bit in bits for _ in range(self.rep)]

        Z = torch.matmul(self.W, F - self.mu)

        nonce = self.derive_nonce(self.key, file_id)

        s_total = torch.tensor(0.0, device=self.device)
        denom = torch.tensor(0.0, device=self.device)

        bit_idx = 0

        for c, (s, e) in enumerate(chunk_ranges(T, self.M)):

            pairs = self._anisotropic_pairs(nonce, c, n, P_eff)
            payload = pairs[self.P0:] if self.P0 > 0 else pairs

            Lc = e - s
            sub = max(1, Lc // self.L)

            num_planes = len(payload)

            # ---- chips once per chunk ----
            chips_all = prf_gaussian(
                self.key,
                b'CHIP' + nonce + struct.pack(">I", c),
                (num_planes, self.L),
                device=self.device
            ).sign()

            for p, (i, j) in enumerate(payload):

                chips = chips_all[p]

                lam_norm = torch.sqrt(self.lam[i] * self.lam[j])

                bit = bits_coded[bit_idx % len(bits_coded)]
                beta = 1.0 if bit == 1 else -1.0

                # collect windows
                Zi_windows = []
                Zj_windows = []
                lengths = []

                for ell in range(self.L):

                    ss = s + ell * sub
                    ee = s + (ell + 1) * sub if ell < self.L - 1 else e

                    if ss >= ee:
                        continue

                    Zi = Z[i, ss:ee]
                    Zj = Z[j, ss:ee]

                    Zi_windows.append(Zi)
                    Zj_windows.append(Zj)
                    lengths.append(ee - ss)

                if not Zi_windows:
                    continue

                Zi_pad = torch.nn.utils.rnn.pad_sequence(Zi_windows, batch_first=True)
                Zj_pad = torch.nn.utils.rnn.pad_sequence(Zj_windows, batch_first=True)

                Zi_pad = Zi_pad - Zi_pad.mean(dim=1, keepdim=True)
                Zj_pad = Zj_pad - Zj_pad.mean(dim=1, keepdim=True)

                cov = torch.sum(Zi_pad * Zj_pad, dim=1)

                s_total += beta * torch.sum(chips[:len(cov)] * cov)

                denom += lam_norm * sum(lengths)

                bit_idx += 1

        return float(s_total / torch.clamp(denom, min=1e-12))

    @torch.no_grad()
    def presence_score_from_waveform(self, enc: EncodecLatent_NEW, y_24k: torch.Tensor, file_id: str,
                                     bits: List[int]) -> float:
        F = enc.encode_test_batch(y_24k)
        return self.presence_score_from_features(F, file_id, bits)

    @torch.no_grad()
    def presence_score_from_path(self, enc: EncodecLatent_NEW, path: str, bits: List[int]) -> float:
        F = enc.encode_test_batch(enc.load_audio(path))
        file_id = os.path.splitext(os.path.basename(path))[0]
        return self.presence_score_from_features(F, file_id, bits)