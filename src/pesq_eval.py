import pesq
import torchaudio.functional as F
import torchmetrics
import torch


# --- PESQ METRIC CLASS ---
class PesqMetricTorch(torchmetrics.Metric):
    """Computes PESQ-WB after resampling to 16kHz."""
    sum_pesq: torch.Tensor
    total: torch.Tensor

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sr = sample_rate
        self.target_sr = 16000
        self.add_state("sum_pesq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def score_batch(self, preds: torch.Tensor, targets: torch.Tensor) -> list[float]:

        # High-quality resampling
        if self.sr != self.target_sr:
            preds = F.resample(preds, self.sr, self.target_sr)
            targets = F.resample(targets, self.sr, self.target_sr)

        batch_scores = []

        for i in range(preds.size(0)):
            ref = targets[i, 0].detach().cpu().numpy()
            deg = preds[i, 0].detach().cpu().numpy()
            try:
                # 'wb' = Wideband mode (standard for 16kHz speech)
                score = pesq.pesq(self.target_sr, ref, deg, 'wb')
                self.sum_pesq += score
                self.total += 1
                batch_scores.append(score)
            except pesq.NoUtterancesError:
                # Handle silence or very short segments
                batch_scores.append(float('nan'))
            except Exception as e:
                print(f"[WARN] PESQ Error: {e}")
                #batch_scores.append(float('nan'))

        return batch_scores

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        pass

    def compute(self):
        return self.sum_pesq / self.total if self.total > 0 else torch.tensor(0.0)