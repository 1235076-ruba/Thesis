"""
Fine-tune SecureBERT for requirement quality classification:
  clear | unclear | incomplete

For small labeled datasets (<100 rows), uses SecureBERT **embeddings** +
a balanced logistic classifier (reliable with ~30 examples).

For larger datasets, optional full end-to-end fine-tuning is available.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LABELS: Tuple[str, ...] = ("clear", "unclear", "incomplete")
DEFAULT_SECUREBERT = "ehsanaghaei/SecureBERT"
FALLBACK_SECUREBERT = "cisco-ai/SecureBERT2.0-base"
EMBEDDING_MODE = "securebert_embedding_lr"
FINETUNE_MODE = "securebert_finetune"
SMALL_DATASET_THRESHOLD = 100

# Must be set before `transformers` is imported (see modeling_utils.py).
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_HF_LOAD_KWARGS = {"use_safetensors": False}


def normalize_label(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    mapping = {
        "clear": "clear",
        "unclear": "unclear",
        "incomplete": "incomplete",
        "in-complete": "incomplete",
    }
    return mapping.get(text)


def detect_classification_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if isinstance(col, str) and col.strip().lower() == "classification":
            return col
    for col in df.columns:
        if isinstance(col, str) and "classif" in col.lower():
            return col
    return None


def detect_requirement_text_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if isinstance(col, str) and "requirement" in col.lower() and "text" in col.lower():
            return col
    for col in df.columns:
        if isinstance(col, str) and "text" in col.lower():
            return col
    return str(df.columns[-1])


@dataclass
class TrainResult:
    model_dir: Path
    train_size: int
    val_size: int
    val_accuracy: float
    val_f1_macro: float
    labels: Tuple[str, ...]
    mode: str


class RequirementClassifier:
    """SecureBERT-based 3-class requirement quality classifier."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._mode = EMBEDDING_MODE
        self._tokenizer = None
        self._encoder = None
        self._sklearn_clf = None
        self._torch_model = None
        self._label2id: Dict[str, int] = {}
        self._id2label: Dict[int, str] = {}
        self._base_model = DEFAULT_SECUREBERT

    @property
    def is_ready(self) -> bool:
        return (self.model_dir / "label_map.json").exists()

    def _load_metadata(self) -> None:
        meta_path = self.model_dir / "label_map.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing label map: {meta_path}")
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        self._label2id = {k: int(v) for k, v in data["label2id"].items()}
        self._id2label = {int(k): v for k, v in data["id2label"].items()}
        self._mode = data.get("mode", FINETUNE_MODE)
        self._base_model = data.get("base_model", DEFAULT_SECUREBERT)

    def _load_encoder(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        import torch

        if self._tokenizer is None:
            enc_path = self.model_dir / "tokenizer_config.json"
            if enc_path.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            else:
                self._tokenizer = _try_load_tokenizer(self._base_model)

        if self._encoder is None:
            self._encoder = AutoModel.from_pretrained(
                self._base_model, **_HF_LOAD_KWARGS
            )
            self._encoder.eval()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._encoder.to(self._device)

    def _embed_texts(self, texts: Sequence[str], batch_size: int = 8) -> np.ndarray:
        import torch

        self._load_encoder()
        assert self._tokenizer is not None and self._encoder is not None

        vectors: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = [str(t).strip() for t in texts[start : start + batch_size]]
                enc = self._tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                out = self._encoder(**enc)
                # Mean pooling over tokens (attention-mask aware)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (out.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = (summed / counts).cpu().numpy()
                vectors.append(pooled)
        return np.vstack(vectors)

    def load(self) -> None:
        self._load_metadata()
        if self._mode == EMBEDDING_MODE:
            import joblib

            sk_path = self.model_dir / "classifier.joblib"
            if not sk_path.exists():
                raise FileNotFoundError(f"Missing classifier: {sk_path}")
            self._sklearn_clf = joblib.load(sk_path)
            self._load_encoder()
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._torch_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir
            )
            self._torch_model.eval()

    def predict(self, texts: Sequence[str]) -> List[str]:
        if self._mode == EMBEDDING_MODE:
            if self._sklearn_clf is None:
                self.load()
            X = self._embed_texts(texts)
            ids = self._sklearn_clf.predict(X)
            return [self._id2label[int(i)] for i in ids]

        import torch

        if self._torch_model is None or self._tokenizer is None:
            self.load()

        cleaned = [str(t).strip() for t in texts]
        if not cleaned:
            return []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_model.to(device)
        self._torch_model.eval()

        predictions: List[str] = []
        with torch.no_grad():
            for start in range(0, len(cleaned), 8):
                batch = cleaned[start : start + 8]
                enc = self._tokenizer(
                    batch, truncation=True, padding=True, max_length=256, return_tensors="pt"
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = self._torch_model(**enc).logits
                ids = logits.argmax(dim=-1).cpu().tolist()
                predictions.extend(self._id2label[i] for i in ids)
        return predictions

    def predict_one(self, text: str) -> str:
        return self.predict([text])[0]


def _resolve_base_model() -> str:
    return os.environ.get("SECUREBERT_MODEL", DEFAULT_SECUREBERT)


def _try_load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return AutoTokenizer.from_pretrained(FALLBACK_SECUREBERT)


def _load_labeled_rows(
    excel_path: Path,
    text_column: Optional[str],
    label_column: Optional[str],
) -> Tuple[List[str], List[str], str, str]:
    df = pd.read_excel(excel_path)
    req_col = text_column or detect_requirement_text_column(df)
    cls_col = label_column or detect_classification_column(df)
    if cls_col is None:
        raise ValueError("No Classification column found in the Excel file.")

    texts: List[str] = []
    labels: List[str] = []
    for _, row in df.iterrows():
        raw = row[req_col]
        if pd.isna(raw) or not str(raw).strip():
            continue
        label = normalize_label(row[cls_col])
        if label is None:
            continue
        texts.append(str(raw).strip())
        labels.append(label)
    return texts, labels, req_col, cls_col


def _train_embedding_classifier(
    texts: List[str],
    labels: List[str],
    model_dir: Path,
    base_model: str,
    seed: int,
) -> TrainResult:
    """
    SecureBERT encoder (frozen) + balanced LogisticRegression.
    Best practice for small requirement datasets (~30–100 rows).
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    label2id = {label: i for i, label in enumerate(LABELS)}
    y = np.array([label2id[l] for l in labels])

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _try_load_tokenizer(base_model)
    tokenizer.save_pretrained(str(model_dir))

    # Embed all labeled requirements once
    temp_clf = RequirementClassifier(model_dir)
    temp_clf._base_model = base_model
    temp_clf._tokenizer = tokenizer
    X = temp_clf._embed_texts(texts)

    sk_clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
        random_state=seed,
        solver="lbfgs",
    )

    # Cross-validated metrics on full labeled set (honest for n=31)
    n_splits = min(5, min(np.bincount(y)))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        cv_preds = cross_val_predict(sk_clf, X, y, cv=cv)
        val_acc = float(accuracy_score(y, cv_preds))
        val_f1 = float(f1_score(y, cv_preds, average="macro"))
        val_size = len(y)
    else:
        val_acc = 0.0
        val_f1 = 0.0
        val_size = 0

    # Fit on ALL labeled data for deployment
    sk_clf.fit(X, y)
    joblib.dump(sk_clf, model_dir / "classifier.joblib")

    meta = {
        "label2id": label2id,
        "id2label": {str(i): label for label, i in label2id.items()},
        "base_model": base_model,
        "labels": list(LABELS),
        "mode": EMBEDDING_MODE,
    }
    (model_dir / "label_map.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    return TrainResult(
        model_dir=model_dir,
        train_size=len(texts),
        val_size=val_size,
        val_accuracy=val_acc,
        val_f1_macro=val_f1,
        labels=LABELS,
        mode=EMBEDDING_MODE,
    )


def _train_finetune_classifier(
    texts: List[str],
    labels: List[str],
    model_dir: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_ratio: float,
    seed: int,
) -> TrainResult:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    import inspect
    import torch
    from torch.utils.data import Dataset

    label2id = {label: i for i, label in enumerate(LABELS)}
    y = np.array([label2id[l] for l in labels])

    try:
        x_train, x_val, y_train, y_val = train_test_split(
            texts, y, test_size=val_ratio, random_state=seed, stratify=y
        )
    except ValueError:
        x_train, x_val, y_train, y_val = train_test_split(
            texts, y, test_size=val_ratio, random_state=seed
        )

    tokenizer = _try_load_tokenizer(base_model)
    id2label = {i: label for label, i in label2id.items()}
    try:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id,
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            FALLBACK_SECUREBERT,
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id,
        )

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weight_tensor = torch.tensor(
        [weights[label2id[id2label[i]]] for i in range(len(LABELS))],
        dtype=torch.float,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_t = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor.to(outputs.logits.device))
            loss = loss_fn(outputs.logits, labels_t)
            return (loss, outputs) if return_outputs else loss

    class ReqDataset(Dataset):
        def __init__(self, x: List[str], y_arr: np.ndarray):
            self.x = x
            self.y = y_arr

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, idx: int):
            enc = tokenizer(self.x[idx], truncation=True, max_length=256)
            enc["labels"] = int(self.y[idx])
            return enc

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        save_total_limit=1,
        report_to=[],
        seed=seed,
    )

    def compute_metrics(eval_pred):
        logits, labels_arr = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels_arr, preds)),
            "f1_macro": float(f1_score(labels_arr, preds, average="macro")),
        }

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": ReqDataset(list(x_train), y_train),
        "eval_dataset": ReqDataset(list(x_val), y_val),
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    meta = {
        "label2id": label2id,
        "id2label": {str(i): label for label, i in label2id.items()},
        "base_model": base_model,
        "labels": list(LABELS),
        "mode": FINETUNE_MODE,
    }
    (model_dir / "label_map.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    val_preds: List[int] = []
    with torch.no_grad():
        for start in range(0, len(x_val), batch_size):
            batch = x_val[start : start + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding=True, max_length=256, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            val_preds.extend(logits.argmax(dim=-1).cpu().tolist())

    val_labels = [id2label[i] for i in y_val]
    val_pred_labels = [id2label[i] for i in val_preds]

    return TrainResult(
        model_dir=model_dir,
        train_size=len(x_train),
        val_size=len(x_val),
        val_accuracy=float(accuracy_score(val_labels, val_pred_labels)),
        val_f1_macro=float(f1_score(val_labels, val_pred_labels, average="macro")),
        labels=LABELS,
        mode=FINETUNE_MODE,
    )


def train_classifier(
    excel_path: Path,
    model_dir: Path,
    *,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    epochs: int = 8,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    val_ratio: float = 0.2,
    seed: int = 42,
    force_finetune: bool = False,
) -> TrainResult:
    texts, labels, _, _ = _load_labeled_rows(excel_path, text_column, label_column)

    if len(texts) < 9:
        raise ValueError(
            f"Need at least 9 labeled requirements to train; found {len(texts)}."
        )

    base_model = _resolve_base_model()

    if len(texts) < SMALL_DATASET_THRESHOLD and not force_finetune:
        return _train_embedding_classifier(texts, labels, model_dir, base_model, seed)

    return _train_finetune_classifier(
        texts,
        labels,
        model_dir,
        base_model,
        epochs,
        batch_size,
        learning_rate,
        val_ratio,
        seed,
    )


def load_classifier(model_dir: Optional[Path] = None) -> RequirementClassifier:
    root = Path(__file__).resolve().parent
    path = model_dir or (root / "models" / "requirement_securebert")
    clf = RequirementClassifier(path)
    if not clf.is_ready:
        raise FileNotFoundError(
            f"No trained model at {path}. Run: python train_classifier.py"
        )
    return clf
