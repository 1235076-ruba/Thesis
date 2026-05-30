"""
Pairwise requirement conflict detection with SecureBERT.

A conflict exists when two requirements contradict each other such that
satisfying both is impossible or very difficult.

Training uses the **Existing conflict** column (target requirement ID).
For small labeled datasets, SecureBERT embeddings + pair features +
balanced logistic regression (same strategy as quality classification).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from classify_requirements import (
    DEFAULT_SECUREBERT,
    _resolve_base_model,
    detect_requirement_text_column,
)

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

CONFLICT_MODE = "securebert_pair_lr"
PAIR_SEP = " [SEP] "


def detect_requirement_id_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if isinstance(col, str) and col.strip().lower() == "requirement id":
            return col
    for col in df.columns:
        if isinstance(col, str) and "requirement" in col.lower() and "id" in col.lower():
            return col
    return None


def detect_conflict_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if isinstance(col, str) and col.strip().lower() == "existing conflict":
            return col
    for col in df.columns:
        if isinstance(col, str) and "conflict" in col.lower():
            return col
    return None


def normalize_req_id(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


@dataclass
class RequirementRecord:
    req_id: str
    text: str
    labeled_conflict_with: Optional[str] = None


@dataclass
class ConflictTrainResult:
    model_dir: Path
    positive_pairs: int
    negative_pairs: int
    train_accuracy: float
    train_f1: float
    threshold: float
    mode: str


@dataclass
class ConflictPrediction:
    req_id: str
    predicted_conflict_with: str
    confidence: float


def _pair_key(id_a: str, id_b: str) -> Tuple[str, str]:
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


def _ordered_pair_features(
    emb_a: np.ndarray, emb_b: np.ndarray, id_a: str, id_b: str
) -> np.ndarray:
    """Build pair features in a stable ID order so scoring is symmetric."""
    if id_a <= id_b:
        return _build_pair_features(emb_a, emb_b)
    return _build_pair_features(emb_b, emb_a)


def load_requirement_records(
    excel_path: Path,
    *,
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    conflict_column: Optional[str] = None,
) -> Tuple[List[RequirementRecord], str, Optional[str], Optional[str]]:
    df = pd.read_excel(excel_path)
    req_col = text_column or detect_requirement_text_column(df)
    id_col = id_column or detect_requirement_id_column(df)
    conflict_col = conflict_column or detect_conflict_column(df)

    if id_col is None:
        raise ValueError("No Requirement ID column found in the Excel file.")

    id_to_row: Dict[str, str] = {}
    records: List[RequirementRecord] = []
    auto_idx = 1

    for _, row in df.iterrows():
        raw_text = row[req_col]
        if pd.isna(raw_text) or not str(raw_text).strip():
            continue

        req_id = normalize_req_id(row[id_col]) if id_col else None
        if not req_id:
            req_id = f"REQ-AUTO-{auto_idx:04d}"
            auto_idx += 1

        text = str(raw_text).strip()
        id_to_row[req_id] = text

        labeled: Optional[str] = None
        if conflict_col and conflict_col in df.columns:
            labeled = normalize_req_id(row[conflict_col])

        records.append(
            RequirementRecord(req_id=req_id, text=text, labeled_conflict_with=labeled)
        )

    return records, req_col, id_col, conflict_col


def extract_labeled_pairs(
    records: Sequence[RequirementRecord],
) -> List[Tuple[str, str, str, str]]:
    """Return (text_a, text_b, id_a, id_b) for each labeled conflict."""
    id_to_text = {r.req_id: r.text for r in records}
    pairs: List[Tuple[str, str, str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    for rec in records:
        if not rec.labeled_conflict_with:
            continue
        other_id = rec.labeled_conflict_with
        if other_id not in id_to_text:
            continue
        key = _pair_key(rec.req_id, other_id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((rec.text, id_to_text[other_id], rec.req_id, other_id))

    return pairs


def _build_pair_features(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Concatenate embeddings and element-wise interaction features."""
    diff = np.abs(emb_a - emb_b)
    prod = emb_a * emb_b
    if emb_a.ndim == 1:
        return np.concatenate([emb_a, emb_b, diff, prod])
    return np.hstack([emb_a, emb_b, diff, prod])


class ConflictDetector:
    """SecureBERT pairwise conflict detector."""

    def __init__(self, model_dir: Path, encoder_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir)
        self.encoder_dir = Path(encoder_dir) if encoder_dir else self.model_dir
        self._embedder = None
        self._clf = None
        self._threshold = 0.5
        self._base_model = DEFAULT_SECUREBERT
        self._mode = CONFLICT_MODE

    @property
    def is_ready(self) -> bool:
        return (self.model_dir / "conflict_classifier.joblib").exists()

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        from classify_requirements import RequirementClassifier

        meta_path = self.model_dir / "conflict_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._base_model = meta.get("base_model", DEFAULT_SECUREBERT)
            self._threshold = float(meta.get("threshold", 0.5))
            enc_dir = meta.get("encoder_dir")
            if enc_dir:
                self.encoder_dir = Path(enc_dir)

        embedder = RequirementClassifier(self.encoder_dir)
        embedder._base_model = self._base_model
        self._embedder = embedder

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load_embedder()
        assert self._embedder is not None
        return self._embedder._embed_texts(texts)

    def _load_classifier(self) -> None:
        if self._clf is not None:
            return
        import joblib

        path = self.model_dir / "conflict_classifier.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing conflict classifier: {path}")
        self._clf = joblib.load(path)
        meta_path = self.model_dir / "conflict_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._threshold = float(meta.get("threshold", 0.5))

    def _pair_matrix(
        self,
        texts_a: Sequence[str],
        texts_b: Sequence[str],
        ids_a: Optional[Sequence[str]] = None,
        ids_b: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        all_texts = list(texts_a) + list(texts_b)
        embs = self._embed(all_texts)
        n = len(texts_a)
        emb_a = embs[:n]
        emb_b = embs[n : n + len(texts_b)]

        if ids_a is None:
            ids_a = [str(i) for i in range(n)]
        if ids_b is None:
            ids_b = [str(i) for i in range(len(texts_b))]

        return np.array(
            [
                _ordered_pair_features(emb_a[i], emb_b[i], ids_a[i], ids_b[i])
                for i in range(n)
            ]
        )

    def predict_pair_proba(self, text_a: str, text_b: str, id_a: str = "a", id_b: str = "b") -> float:
        self._load_classifier()
        X = self._pair_matrix([text_a], [text_b], [id_a], [id_b])
        assert self._clf is not None
        return float(self._clf.predict_proba(X)[0, 1])

    def predict_pairs(
        self, records: Sequence[RequirementRecord]
    ) -> Dict[str, ConflictPrediction]:
        """For each requirement, predict the most likely conflicting partner (if any)."""
        self._load_classifier()
        if len(records) < 2:
            return {}

        ids = [r.req_id for r in records]
        texts = [r.text for r in records]
        embs = self._embed(texts)
        n = len(records)

        best: Dict[str, ConflictPrediction] = {}
        assert self._clf is not None

        for i in range(n):
            if n == 1:
                break
            others = [j for j in range(n) if j != i]
            feats = np.array(
                [
                    _ordered_pair_features(embs[i], embs[j], ids[i], ids[j])
                    for j in others
                ]
            )
            probs = self._clf.predict_proba(feats)[:, 1]
            ranked = sorted(zip(probs, others), reverse=True)
            best_prob, best_j = ranked[0]
            second_prob = ranked[1][0] if len(ranked) > 1 else 0.0
            margin = float(best_prob - second_prob)
            if best_prob >= self._threshold and margin >= 0.04:
                partner = ids[best_j]
                best[ids[i]] = ConflictPrediction(
                    req_id=ids[i],
                    predicted_conflict_with=partner,
                    confidence=float(best_prob),
                )

        return best

    def predict_for_record(
        self, record: RequirementRecord, all_records: Sequence[RequirementRecord]
    ) -> Tuple[str, float]:
        """Return (conflict_id or '', confidence) for one requirement."""
        predictions = self.predict_pairs(all_records)
        hit = predictions.get(record.req_id)
        if hit is None:
            return "", 0.0
        return hit.predicted_conflict_with, hit.confidence


def _sample_negative_pairs(
    records: Sequence[RequirementRecord],
    positive_keys: Set[Tuple[str, str]],
    *,
    max_negatives: int,
    seed: int,
) -> List[Tuple[str, str, str, str]]:
    rng = random.Random(seed)
    ids = [r.req_id for r in records]
    id_to_text = {r.req_id: r.text for r in records}
    candidates: List[Tuple[str, str, str, str]] = []

    for i, id_a in enumerate(ids):
        for id_b in ids[i + 1 :]:
            if _pair_key(id_a, id_b) in positive_keys:
                continue
            candidates.append((id_to_text[id_a], id_to_text[id_b], id_a, id_b))

    rng.shuffle(candidates)
    return candidates[:max_negatives]


def train_conflict_detector(
    excel_path: Path,
    model_dir: Path,
    *,
    encoder_dir: Optional[Path] = None,
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    conflict_column: Optional[str] = None,
    seed: int = 42,
    negative_ratio: int = 30,
) -> ConflictTrainResult:
    """
    Train pairwise conflict detector from **Existing conflict** labels.

    Each labeled row defines one positive pair (this requirement, target ID).
    Negative pairs are sampled from all other requirement combinations.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    records, _, _, _ = load_requirement_records(
        excel_path,
        text_column=text_column,
        id_column=id_column,
        conflict_column=conflict_column,
    )

    positives = extract_labeled_pairs(records)
    if not positives:
        raise ValueError(
            "No labeled conflict pairs found. Fill the 'Existing conflict' column "
            "with target Requirement IDs."
        )

    positive_keys = {_pair_key(id_a, id_b) for _, _, id_a, id_b in positives}
    max_neg = max(50, len(positives) * negative_ratio)
    negatives = _sample_negative_pairs(
        records, positive_keys, max_negatives=max_neg, seed=seed
    )

    texts_a = [p[0] for p in positives] + [n[0] for n in negatives]
    texts_b = [p[1] for p in positives] + [n[1] for n in negatives]
    ids_a = [p[2] for p in positives] + [n[2] for n in negatives]
    ids_b = [p[3] for p in positives] + [n[3] for n in negatives]
    y = np.array([1] * len(positives) + [0] * len(negatives))

    base_model = _resolve_base_model()
    root = Path(__file__).resolve().parent
    resolved_encoder_dir = encoder_dir or (root / "models" / "requirement_securebert")
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    detector = ConflictDetector(model_dir, encoder_dir=resolved_encoder_dir)
    X = detector._pair_matrix(texts_a, texts_b, ids_a, ids_b)

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(X, y)

    train_probs = clf.predict_proba(X)[:, 1]
    train_preds = (train_probs >= 0.5).astype(int)
    train_acc = float(accuracy_score(y, train_preds))
    train_f1 = float(f1_score(y, train_preds, zero_division=0))

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        preds = (train_probs >= threshold).astype(int)
        score = f1_score(y, preds, zero_division=0)
        if score >= best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    # With few labeled conflicts, keep threshold high to limit false positives.
    pos_probs = train_probs[y == 1]
    min_pos = float(np.min(pos_probs)) if len(pos_probs) else 0.75
    best_threshold = max(0.75, min(best_threshold, min_pos - 0.05))

    joblib.dump(clf, model_dir / "conflict_classifier.joblib")
    meta = {
        "mode": CONFLICT_MODE,
        "base_model": base_model,
        "encoder_dir": str(resolved_encoder_dir.resolve()),
        "threshold": best_threshold,
        "positive_pairs": len(positives),
        "negative_pairs": len(negatives),
    }
    (model_dir / "conflict_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    return ConflictTrainResult(
        model_dir=model_dir,
        positive_pairs=len(positives),
        negative_pairs=len(negatives),
        train_accuracy=train_acc,
        train_f1=best_f1,
        threshold=best_threshold,
        mode=CONFLICT_MODE,
    )


def load_conflict_detector(
    model_dir: Optional[Path] = None,
    encoder_dir: Optional[Path] = None,
) -> ConflictDetector:
    root = Path(__file__).resolve().parent
    path = model_dir or (root / "models" / "requirement_conflict_securebert")
    detector = ConflictDetector(path, encoder_dir=encoder_dir)
    if not detector.is_ready:
        raise FileNotFoundError(
            f"No trained conflict model at {path}. Run: python train_conflict_detector.py"
        )
    return detector
