"""
Weinzierl et al. (2020) - Prescriptive BPM: Recommending Next Best Actions.
arXiv:2008.08693

Offline component:
  mpp  -- multi-task LSTM: predicts next activity (classification) +
          KPI value/duration (regression) for a running process instance.
  mcs  -- k-NN suffix store: finds k historical suffixes closest (L2 on
          ordinal-encoded, zero-padded sequences) to the predicted suffix.

Online component:
  1. Extract prefix (completed activities) from bitmask state.
  2. Predict the complete suffix iteratively with mpp.
  3. Sum prefix + suffix KPI; if total > threshold, trigger prescription.
  4. Query mcs for k nearest candidates, rank by ascending KPI.
  5. Return the first activity of the lowest-KPI valid candidate
     (validated against the action mask — our substitute for BPS).
  6. Fallback chain: LSTM most-likely next → random valid action.
"""

import json
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Neural network ─────────────────────────────────────────────────────────────

class _MultiTaskLSTM(nn.Module):
    """Shared LSTM encoder + two output heads (activity clf + KPI regression)."""

    def __init__(self, input_size: int, hidden_size: int, num_activities: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.act_head = nn.Linear(hidden_size, num_activities + 1)  # +1 END token
        self.kpi_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        h = out[:, -1, :]                       # last hidden state
        return self.act_head(h), self.kpi_head(h).squeeze(-1)


# ── Agent ──────────────────────────────────────────────────────────────────────

class NextBestActionAgent:
    """
    Prescriptive agent (Weinzierl et al., 2020) adapted to the health-check
    simulation environment.

    KPI = patient activity duration (minutes); lower is better.
    Threshold t = average throughput time computed from the training event log.
    BPS conformance check = action mask (valid activities at current prefix).
    """

    N_ACT = 21
    END_TOKEN = 21          # sentinel index signalling process completion

    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 0,
        hidden_size: int = 100,
        k_neighbors: int = 5,
        lstm_epochs: int = 50,
        batch_size: int = 256,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.hidden_size = hidden_size
        self.k_neighbors = k_neighbors
        self.lstm_epochs = lstm_epochs
        self.batch_size = batch_size

        # Populated by train() or load()
        self._mpp: Optional[_MultiTaskLSTM] = None
        self._suffix_acts: Optional[List[List[int]]] = None   # historical suffixes
        self._suffix_kpis: Optional[List[float]] = None       # total KPI per suffix
        self._suffix_firsts: Optional[List[int]] = None       # first activity of each suffix
        self._suffix_matrix: Optional[np.ndarray] = None      # padded matrix for L2 kNN
        self.threshold: Optional[float] = None
        self._max_prefix_len: int = 22
        self._max_suffix_len: int = 22
        self._device: str = "cpu"

        self._load_activity_info()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _load_activity_info(self):
        for path in [
            os.path.join("data", "raw", "activity_info.json"),
            os.path.join("..", "data", "raw", "activity_info.json"),
        ]:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    info = json.load(f)
                break
        else:
            raise FileNotFoundError("activity_info.json not found")

        self._name_to_id: dict = {name: d["id"] - 1 for name, d in info.items()}
        self._mean_times = np.zeros(self.N_ACT, dtype=np.float32)
        for name, d in info.items():
            idx = d["id"] - 1
            mt = d["mean_time"]
            self._mean_times[idx] = (
                (mt["Cash"] + mt["Credit"]) / 2.0 if isinstance(mt, dict) else float(mt)
            )

    # ── Offline component ──────────────────────────────────────────────────────

    def train(self, event_log_path: str):
        """Full offline training: build mpp and mcs from a random-base event log."""
        import pandas as pd

        print("📊 Parsing event log for NextBestAction...")
        df = pd.read_csv(event_log_path)
        cases = self._parse_cases(df)
        print(f"   {len(cases)} valid cases")

        print("📊 Building LSTM training samples...")
        samples = self._build_lstm_samples(cases)
        print(f"   {len(samples)} (prefix, next_act, kpi) samples")

        print("📊 Training multi-task LSTM (mpp)...")
        self._mpp = self._train_lstm(samples)
        self._mpp.eval()

        print("📊 Building suffix database (mcs)...")
        self._build_suffix_db(cases)
        print(f"   {len(self._suffix_acts)} suffix entries")

        self.threshold = float(np.mean([sum(d) for _, d in cases]))
        print(f"   KPI threshold: {self.threshold:.2f} mins")

    def _parse_cases(self, df) -> List[Tuple[List[int], List[float]]]:
        """Parse event log into [(activities, durations), ...] per case."""
        cases = []
        for _, grp in df.groupby("CaseID"):
            grp = grp.sort_values("Timestamp")
            starts: dict = {}
            acts, durs = [], []
            for _, row in grp.iterrows():
                act = row["Activity"]
                lc = row["Lifecycle"]
                ts = float(row["Timestamp"])
                if lc == "START":
                    starts[act] = ts
                elif lc == "COMPLETE" and act in starts:
                    dur = max(0.0, ts - starts.pop(act))
                    idx = self._name_to_id.get(act)
                    if idx is not None:
                        acts.append(int(idx))
                        durs.append(float(dur))
            if len(acts) >= 2:
                cases.append((acts, durs))
        return cases

    def _build_lstm_samples(
        self, cases: List[Tuple[List[int], List[float]]]
    ) -> List[Tuple[List[int], int, float]]:
        """Extract (prefix, next_activity, kpi_value) training tuples."""
        samples = []
        for acts, durs in cases:
            acts_ext = acts + [self.END_TOKEN]
            durs_ext = durs + [0.0]
            for k in range(1, len(acts)):
                samples.append((acts_ext[:k], int(acts_ext[k]), float(durs_ext[k])))
        if samples:
            self._max_prefix_len = max(len(s[0]) for s in samples)
            raw_kpis = np.array([s[2] for s in samples], dtype=np.float32)
            self._kpi_mean = float(raw_kpis.mean())
            self._kpi_std = float(raw_kpis.std()) if raw_kpis.std() > 0 else 1.0
        else:
            self._kpi_mean = 0.0
            self._kpi_std = 1.0
        return samples

    def _train_lstm(
        self, samples: List[Tuple[List[int], int, float]]
    ) -> _MultiTaskLSTM:
        input_size = self.N_ACT + 1  # +1 for END token
        model = _MultiTaskLSTM(input_size, self.hidden_size, self.N_ACT)
        model.to(self._device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        n = len(samples)
        for epoch in range(self.lstm_epochs):
            np.random.shuffle(samples)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, n, self.batch_size):
                batch = samples[i : i + self.batch_size]
                x, acts_t, kpis_t = self._collate(batch)
                optimizer.zero_grad()
                act_logits, kpi_pred = model(x)
                # Normalize KPI targets to unit scale before MSE
                kpis_norm = (kpis_t - self._kpi_mean) / self._kpi_std
                loss = ce_loss(act_logits, acts_t) + 0.01 * mse_loss(kpi_pred, kpis_norm)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 10 == 0:
                avg = epoch_loss / max(1, n_batches)
                print(f"   Epoch {epoch + 1}/{self.lstm_epochs} | loss={avg:.4f}")
        return model

    def _collate(
        self, batch: List[Tuple[List[int], int, float]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad prefix sequences and stack into tensors."""
        B = len(batch)
        input_size = self.N_ACT + 1
        ml = self._max_prefix_len
        X = np.zeros((B, ml, input_size), dtype=np.float32)
        acts_out = np.zeros(B, dtype=np.int64)
        kpis_out = np.zeros(B, dtype=np.float32)
        for i, (prefix, a, kpi) in enumerate(batch):
            for j, pa in enumerate(prefix[-ml:]):
                X[i, j, pa] = 1.0
            acts_out[i] = a
            kpis_out[i] = kpi
        return (
            torch.FloatTensor(X).to(self._device),
            torch.LongTensor(acts_out).to(self._device),
            torch.FloatTensor(kpis_out).to(self._device),
        )

    def _build_suffix_db(self, cases: List[Tuple[List[int], List[float]]]):
        """Build suffix store: padded matrix + per-suffix metadata."""
        suf_acts, suf_kpis, suf_firsts = [], [], []
        for acts, durs in cases:
            for k in range(1, len(acts)):
                s = acts[k:]
                suf_acts.append(s)
                suf_kpis.append(float(sum(durs[k:])))
                suf_firsts.append(int(s[0]))

        self._suffix_acts = suf_acts
        self._suffix_kpis = suf_kpis
        self._suffix_firsts = suf_firsts

        if suf_acts:
            ml = max(len(s) for s in suf_acts)
            self._max_suffix_len = ml
            mat = np.zeros((len(suf_acts), ml), dtype=np.float32)
            for i, s in enumerate(suf_acts):
                mat[i, : len(s)] = [a + 1 for a in s]  # 1-indexed; 0 = padding
            self._suffix_matrix = mat

    # ── Online component ───────────────────────────────────────────────────────

    def _predict_next(self, prefix_acts: List[int]) -> Tuple[int, float]:
        """Single-step LSTM prediction: (next_activity, kpi_value in original minutes)."""
        input_size = self.N_ACT + 1
        ml = self._max_prefix_len
        X = np.zeros((1, ml, input_size), dtype=np.float32)
        for j, a in enumerate(prefix_acts[-ml:]):
            X[0, j, a] = 1.0
        x = torch.FloatTensor(X).to(self._device)
        with torch.no_grad():
            act_logits, kpi_pred = self._mpp(x)
        # Denormalize back to original minutes scale
        kpi_minutes = float(kpi_pred[0].item()) * self._kpi_std + self._kpi_mean
        return int(torch.argmax(act_logits[0]).item()), kpi_minutes

    def _predict_suffix(
        self, prefix_acts: List[int]
    ) -> Tuple[List[int], List[float]]:
        """Iteratively predict the complete suffix (activities + KPI values)."""
        suffix_acts, suffix_kpis = [], []
        done = set(prefix_acts)
        cur = list(prefix_acts)
        for _ in range(self.N_ACT):
            a, kpi = self._predict_next(cur)
            if a == self.END_TOKEN or a in done:
                break
            suffix_acts.append(a)
            suffix_kpis.append(max(0.0, kpi))
            done.add(a)
            cur.append(a)
        return suffix_acts, suffix_kpis

    def _knn_query(self, query_suffix: List[int], k: int) -> np.ndarray:
        """Return indices of k nearest suffix entries (brute-force L2)."""
        ml = self._max_suffix_len
        q = np.zeros((1, ml), dtype=np.float32)
        for i, a in enumerate(query_suffix[:ml]):
            q[0, i] = a + 1  # 1-indexed to match database encoding
        dists = np.linalg.norm(self._suffix_matrix - q, axis=1)
        return np.argsort(dists)[:k]

    def _find_best_candidate(
        self, pred_suffix: List[int], mask: np.ndarray, done: set
    ) -> Optional[int]:
        """
        Find the first-activity of the best valid candidate suffix.
        Ranks candidates by ascending KPI (Sec. 3.2 of paper).
        Validates against action mask (our substitute for BPS).
        """
        k = min(self.k_neighbors, len(self._suffix_acts) if self._suffix_acts else 0)
        if k == 0:
            return None

        neighbor_idx = self._knn_query(pred_suffix, k)
        # Sort by KPI ascending (lower throughput time = better)
        ranked = sorted(neighbor_idx, key=lambda i: self._suffix_kpis[i])

        for i in ranked:
            first_act = self._suffix_firsts[i]
            if first_act not in done and mask[first_act] == 1.0:
                return int(first_act)
        return None

    def act(self, state: np.ndarray, mask: np.ndarray, eps: float = 0.0) -> int:
        """Online prescriptive action selection."""
        valid = np.where(mask == 1.0)[0]
        if len(valid) == 0:
            return 0

        if self._mpp is None:
            return int(np.random.choice(valid))

        # Reconstruct prefix from bitmask (canonical sorted order)
        bits = state[2 : 2 + self.N_ACT]
        prefix = sorted(i for i in range(self.N_ACT) if bits[i] > 0.5)
        done = set(prefix)

        # Paper: skip suffix prediction for prefix size <= 1
        if len(prefix) <= 1:
            a, _ = self._predict_next(prefix if prefix else [0])
            if a < self.N_ACT and mask[a] == 1.0 and a not in done:
                return int(a)
            return int(valid[0])

        # Predict complete suffix
        suf_acts, suf_kpis = self._predict_suffix(prefix)

        # Total predicted KPI = prefix KPI (mean times) + suffix KPI
        prefix_kpi = float(sum(self._mean_times[a] for a in prefix))
        total_kpi = prefix_kpi + sum(suf_kpis)

        # Prescriptive path: KPI exceeds threshold → find best candidate
        if total_kpi > self.threshold and suf_acts:
            best = self._find_best_candidate(suf_acts, mask, done)
            if best is not None:
                return best

        # Default path: most likely next activity from LSTM
        a, _ = self._predict_next(prefix)
        if a < self.N_ACT and mask[a] == 1.0 and a not in done:
            return int(a)

        # Final fallback: first valid action
        return int(valid[0])

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        data = {
            "mpp_state": self._mpp.state_dict() if self._mpp else None,
            "mpp_config": {"hidden_size": self.hidden_size},
            "suffix_acts": self._suffix_acts,
            "suffix_kpis": self._suffix_kpis,
            "suffix_firsts": self._suffix_firsts,
            "suffix_matrix": self._suffix_matrix,
            "threshold": self.threshold,
            "max_prefix_len": self._max_prefix_len,
            "max_suffix_len": self._max_suffix_len,
            "k_neighbors": self.k_neighbors,
            "kpi_mean": getattr(self, "_kpi_mean", 0.0),
            "kpi_std": getattr(self, "_kpi_std", 1.0),
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        cfg = data["mpp_config"]
        self._mpp = _MultiTaskLSTM(self.N_ACT + 1, cfg["hidden_size"], self.N_ACT)
        self._mpp.load_state_dict(data["mpp_state"])
        self._mpp.eval()
        self._device = "cpu"

        self._suffix_acts = data["suffix_acts"]
        self._suffix_kpis = data["suffix_kpis"]
        self._suffix_firsts = data["suffix_firsts"]
        self._suffix_matrix = data["suffix_matrix"]
        self.threshold = data["threshold"]
        self._max_prefix_len = data["max_prefix_len"]
        self._max_suffix_len = data["max_suffix_len"]
        self.k_neighbors = data["k_neighbors"]
        self._kpi_mean = data.get("kpi_mean", 0.0)
        self._kpi_std = data.get("kpi_std", 1.0)
