"""
Clean and segment software requirements from an Excel file.

Segmentation is automatic and spaCy-driven:
  1) Sentence boundaries (doc.sents), skipping splits inside quoted UI text.
  2) Coordinated verb clauses (dependency conj + VERB), including nested patterns
     (must be able to X and Y; matrix verb + advcl with coordinated verb).
  3) Segment validation — invalid splits are rejected and the requirement stays atomic.

Input:  .xlsx / .xls with requirement text (+ optional Classification labels for training)
Output: CSV with columns: requirement_id, requirement, clean, segmentation,
  classification, existing_conflict, predicted_conflict, conflict_confidence

Install:
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
  python train_classifier.py Requirement_DS.xlsx

Usage:
  python process_requirements.py your.xlsx
  python process_requirements.py your.xlsx -o output.csv -c "Requirement text"
  python process_requirements.py your.xlsx --train-classifier
"""

from __future__ import annotations

import os

# Quiet HuggingFace background safetensors conversion (403 on SecureBERT repo).
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse
from datetime import datetime
import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Token
except ImportError:
    print("Install dependencies: pip install -r requirements.txt", file=sys.stderr)
    raise

MODALS = frozenset({"shall", "must", "should", "will", "can", "may", "might"})


def load_nlp() -> Language:
    """Load spaCy English model."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print(
            "spaCy model not found. Run:\n  python -m spacy download en_core_web_sm",
            file=sys.stderr,
        )
        raise


def clean_text(text: object) -> str:
    """
    Normalize requirement text:
    - lowercase
    - remove line-break hyphenation (in- formation -> information)
    - strip unwanted punctuation/symbols
    - collapse whitespace
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    raw = str(text)
    raw = re.sub(r"-\s*\n\s*", "", raw)
    raw = raw.replace("\n", " ").replace("\r", " ")

    lowered = raw.lower()
    lowered = re.sub(r"([a-z])-([a-z])", r"\1\2", lowered)
    cleaned = re.sub(r"[^a-z0-9\s.,;:()\-]", " ", lowered)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_segment(segment: str) -> str:
    return re.sub(r"\s+", " ", segment).strip(" ,.;")


def _subtree_text(token: Token) -> str:
    return " ".join(t.text for t in sorted(token.subtree, key=lambda x: x.i)).strip()


def _tokens_to_text(tokens: List[Token]) -> str:
    if not tokens:
        return ""
    ordered = sorted(set(tokens), key=lambda t: t.i)
    return " ".join(t.text for t in ordered).strip()


def _subject_phrase_tokens(doc: Doc, verb: Token) -> List[Token]:
    tokens: List[Token] = []
    for child in verb.children:
        if child.dep_ not in ("nsubj", "nsubjpass", "csubj", "expl"):
            continue
        for t in child.subtree:
            tokens.append(t)
        for t in doc:
            if t.i < verb.i and t.head in list(child.subtree):
                tokens.append(t)
    return sorted(set(tokens), key=lambda t: t.i)


def _obligation_subject_tokens(doc: Doc, verb: Token) -> List[Token]:
    """Subject for the obligation phrase, walking up from xcomp/conj verbs if needed."""
    tokens = _subject_phrase_tokens(doc, verb)
    if tokens:
        return tokens
    for anc in verb.ancestors:
        if anc.pos_ in ("VERB", "AUX") or anc.dep_ == "ROOT":
            tokens = _subject_phrase_tokens(doc, anc)
            if tokens:
                return tokens
    return []


def _has_modal_in_clause(verb: Token) -> bool:
    """True if this verb phrase is within an obligation modal (must/shall/...)."""
    for t in [verb, *verb.ancestors]:
        if t.pos_ not in ("VERB", "AUX"):
            continue
        for child in t.children:
            if child.dep_ in ("aux", "auxpass") and child.text.lower() in MODALS:
                return True
    return False


def _primary_subject_lemma(verb: Token) -> Optional[str]:
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
            return child.lemma_
    if verb.dep_ == "conj" and verb.head.pos_ == "VERB":
        return _primary_subject_lemma(verb.head)
    return None


def _cc_for_conj(head: Token, conj: Token) -> Optional[Token]:
    ccs = [t for t in head.children if t.dep_ == "cc" and t.i < conj.i]
    return ccs[-1] if ccs else None


def _should_block_sentence_split(original: str) -> bool:
    """Block sentence splits when UI message text is embedded in quotes."""
    if '"' not in original:
        return False
    lowered = original.lower()
    if "stating" not in lowered and "message" not in lowered:
        return False
    # Requirement embeds UI copy; treat period inside quotes as non-splitting
    return len(list(re.finditer(r"\.\s+", original))) > 0 and '"' in original


def is_valid_requirement_segment(segment: str, nlp: Language) -> bool:
    """
    A valid segment must look like a standalone requirement:
    - contains an obligation modal
    - contains a lexical verb
    - has enough tokens
    - is not a bare message fragment (no modal)
    """
    segment = _normalize_segment(segment)
    if len(segment.split()) < 3:
        return False

    doc = nlp(segment)
    has_modal = any(t.text.lower() in MODALS for t in doc)
    has_verb = any(t.pos_ == "VERB" and t.dep_ in ("ROOT", "xcomp", "ccomp", "conj") for t in doc)
    if not (has_modal and has_verb):
        return False

    # Reject orphan continuations (e.g. "try again, without crashing")
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is not None and root.pos_ != "VERB" and not has_modal:
        return False
    if segment.split()[0] in {"try", "please", "again", "without"}:
        return False

    return True


def validate_segments(segments: List[str], nlp: Language) -> bool:
    return len(segments) > 1 and all(is_valid_requirement_segment(s, nlp) for s in segments)


def is_nominal_coordination_only(doc: Doc) -> bool:
    """
    True when coordination links nouns/adjectives/lists only (not independent verb actions).
    Examples: pdf and csv, beautiful and modern, address payment and review.
    """
    for token in doc:
        if token.dep_ != "cc" or token.text.lower() not in {"and", "or", "but"}:
            continue
        head = token.head
        if head.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"}:
            return True
    for token in doc:
        if token.pos_ == "VERB":
            if any(c.dep_ == "conj" and c.pos_ == "VERB" for c in token.children):
                return False
    return False


def propagate_shared_suffix(segments: List[str]) -> List[str]:
    """
    Distribute global trailing constraints (performance, environment, style)
    from the last segment to earlier segments that share the same subject prefix.
    """
    if len(segments) < 2:
        return segments

    m = re.search(
        r"\s+((?:within|under|in the|without|utilizing|according to)\b.+)$",
        segments[-1],
        flags=re.IGNORECASE,
    )
    if not m:
        return segments

    suffix = _normalize_segment(m.group(1))
    prefix = " ".join(segments[0].split()[:3]).lower()

    out: List[str] = []
    for seg in segments:
        if suffix.lower() in seg.lower():
            out.append(seg)
        elif seg.lower().startswith(prefix):
            out.append(_normalize_segment(f"{seg} {suffix}"))
        else:
            out.append(seg)
    return out


def _auxiliary_context_tokens(
    doc: Doc, head: Token, subj_tokens: List[Token]
) -> List[Token]:
    """Extract modals/auxiliary chains tied specifically to the head verb phrase."""
    subj_ids = {t.i for t in subj_tokens}
    tokens: List[Token] = []

    # 1. Capture direct auxiliary modifications of the head verb
    for t in head.children:
        if t.i in subj_ids:
            continue
        if t.dep_ in ("aux", "auxpass", "mark") or t.text.lower() in MODALS:
            tokens.append(t)
        elif t.lemma_ == "be" or t.text.lower() == "able":
            tokens.append(t)
            for grandchild in t.children:
                if grandchild.dep_ in ("aux", "mark") or grandchild.text.lower() in ("to", "be"):
                    tokens.append(grandchild)

    # 2. Ascend to ancestors strictly tracking the immediate modal auxiliary chain
    chain_words = MODALS | {"be", "able", "to"}
    for ancestor in head.ancestors:
        if ancestor.i in subj_ids:
            continue
        if ancestor.pos_ not in ("AUX", "VERB", "ADJ"):
            continue
        # Ensure ancestor is positioned prior to the head action and matches modal properties
        if ancestor.i < head.i and ancestor.text.lower() in chain_words:
            tokens.append(ancestor)
        for t in ancestor.children:
            if t.i >= head.i or t.i in subj_ids:
                continue
            if t.dep_ in ("aux", "auxpass", "mark", "acomp") or t.text.lower() in chain_words:
                tokens.append(t)

    return sorted(set(tokens), key=lambda x: x.i)


def split_by_sentences(doc: Doc, original: str, nlp: Language) -> List[str]:
    if _should_block_sentence_split(original):
        return []
    segments = [_normalize_segment(s.text) for s in doc.sents]
    segments = [s for s in segments if s]
    if len(segments) <= 1:
        return []
    if validate_segments(segments, nlp):
        return segments
    return []


def split_coordination(
    doc: Doc, head: Token, conj_verbs: List[Token], nlp: Language
) -> List[str]:
    """Split coordinated verb phrases and safely reconstruct isolated segments."""
    cc_conj_pairs = [
        (cc, conj)
        for conj in conj_verbs
        if (cc := _cc_for_conj(head, conj)) is not None
    ]
    if not cc_conj_pairs:
        return []

    different_subjects = any(
        _primary_subject_lemma(head) != _primary_subject_lemma(conj)
        and _primary_subject_lemma(conj) is not None
        for _, conj in cc_conj_pairs
    )

    segments: List[str] = []

    if different_subjects:
        for i, (cc, conj) in enumerate(cc_conj_pairs):
            if i == 0:
                segments.append(_normalize_segment(doc[: cc.i].text))
            end = (
                len(doc)
                if conj == cc_conj_pairs[-1][1]
                else cc_conj_pairs[i + 1][0].i
            )
            segments.append(_normalize_segment(doc[conj.left_edge.i : end].text))
    else:
        subj_tokens = _obligation_subject_tokens(doc, head)
        subj_text = _tokens_to_text(subj_tokens)
        aux_tokens = _auxiliary_context_tokens(doc, head, subj_tokens)
        aux_text = _tokens_to_text(aux_tokens)

        first_cc = cc_conj_pairs[0][0]
        segments.append(_normalize_segment(doc[: first_cc.i].text))

        for _, conj in cc_conj_pairs:
            phrase = _subtree_text(conj)
            if aux_text and aux_text.lower() not in phrase.lower():
                if subj_text and subj_text.lower() not in phrase.lower():
                    seg = _normalize_segment(f"{subj_text} {aux_text} {phrase}")
                else:
                    seg = _normalize_segment(f"{aux_text} {phrase}")
            elif subj_text and subj_text.lower() not in phrase.lower():
                seg = _normalize_segment(f"{subj_text} {phrase}")
            else:
                seg = _normalize_segment(phrase)
            segments.append(seg)

    segments = [s for s in segments if s and len(s.split()) >= 3]
    if len(segments) > 1 and all(is_valid_requirement_segment(s, nlp) for s in segments):
        return propagate_shared_suffix(segments)
    return []


def split_advcl_coordination(doc: Doc, nlp: Language) -> List[str]:
    """
    Matrix verb + adverbial clause whose verb has a coordinated sibling.
    e.g. must send ... when user completes purchase and update inventory.
    """
    for root in doc:
        if root.dep_ != "ROOT" or root.pos_ != "VERB":
            continue
        if not _has_modal_in_clause(root):
            continue
        for advcl in root.children:
            if advcl.dep_ != "advcl" or advcl.pos_ != "VERB":
                continue
            conj_verbs = [
                c for c in advcl.children if c.dep_ == "conj" and c.pos_ == "VERB"
            ]
            if not conj_verbs:
                continue
            cc = _cc_for_conj(advcl, conj_verbs[0])
            if cc is None:
                continue
            seg1 = _normalize_segment(doc[: cc.i].text)
            subj_tokens = _obligation_subject_tokens(doc, root)
            subj_text = _tokens_to_text(subj_tokens)
            aux_text = _tokens_to_text(_auxiliary_context_tokens(doc, root, subj_tokens))
            segments = [seg1]
            for conj in conj_verbs:
                phrase = _subtree_text(conj)
                if aux_text and aux_text.lower() not in phrase.lower():
                    if subj_text and subj_text.lower() not in phrase.lower():
                        seg = _normalize_segment(f"{subj_text} {aux_text} {phrase}")
                    else:
                        seg = _normalize_segment(f"{aux_text} {phrase}")
                elif subj_text and subj_text.lower() not in phrase.lower():
                    seg = _normalize_segment(f"{subj_text} {phrase}")
                else:
                    seg = _normalize_segment(phrase)
                segments.append(seg)
            if len(segments) > 1 and all(
                is_valid_requirement_segment(s, nlp) for s in segments
            ):
                return propagate_shared_suffix(segments)
    return []


def segment_requirement(text: str, nlp: Language) -> List[str]:
    """
    Automatic segmentation pipeline (spaCy structure + validation gates).
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    original = str(text).strip()
    doc = nlp(cleaned)

    if is_nominal_coordination_only(doc):
        return [cleaned]

    candidates: List[List[str]] = []

    sent_segs = split_by_sentences(doc, original, nlp)
    if sent_segs:
        candidates.append(sent_segs)

    for head in doc:
        if head.pos_ != "VERB" or not _has_modal_in_clause(head):
            continue
        if head.dep_ == "conj":
            continue
        conj_verbs = [
            c for c in head.children if c.dep_ == "conj" and c.pos_ == "VERB"
        ]
        if conj_verbs:
            coord = split_coordination(doc, head, conj_verbs, nlp)
            if coord:
                candidates.append(coord)

    advcl = split_advcl_coordination(doc, nlp)
    if advcl:
        candidates.append(advcl)

    if not candidates:
        return [cleaned]

    # Prefer the split with exactly two well-formed segments (typical atomic pair)
    candidates.sort(key=lambda segs: (-min(len(s) for s in segs), len(segs)))
    return candidates[0]


def format_segmentation(segments: List[str]) -> str:
    if len(segments) <= 1:
        return ""
    return " || ".join(f"{i}) {s}" for i, s in enumerate(segments, start=1))


def _is_id_column(name: str) -> bool:
    n = name.strip().lower().replace(" ", "_")
    id_markers = ("_id", " id", "id_", "number", "num", "code", "key", "index", "no")
    if n in ("id", "req_id", "requirement_id"):
        return True
    return any(m in n for m in id_markers) and "text" not in n and "description" not in n


def _avg_text_length(series: pd.Series) -> float:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return 0.0
    return float(values.str.len().mean())


def detect_requirement_column(df: pd.DataFrame) -> str:
    normalized = {
        str(c): str(c).strip().lower().replace(" ", "_") for c in df.columns
    }
    for c, n in normalized.items():
        if n in ("requirement_ds", "requirement_text", "requirementtext"):
            return c
    for c, n in normalized.items():
        if "requirement_ds" in n or n.endswith("_text") or n == "requirement_text":
            return c
    for c in df.columns:
        if isinstance(c, str) and "requirement" in c.lower() and "text" in c.lower():
            return c
    name_candidates = [
        c
        for c in df.columns
        if isinstance(c, str)
        and not _is_id_column(c)
        and any(
            k in c.lower()
            for k in ("requirement", "req", "description", "statement", "text")
        )
    ]
    if name_candidates:
        return max(name_candidates, key=lambda col: _avg_text_length(df[col]))
    object_cols = [c for c in df.columns if df[c].dtype == object and not _is_id_column(str(c))]
    if object_cols:
        best = max(object_cols, key=lambda col: _avg_text_length(df[col]))
        if _avg_text_length(df[best]) >= 20:
            return best
    return df.columns[0]


def read_excel(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        return pd.read_excel(path, engine="xlrd")
    return pd.read_excel(path, engine="openpyxl")


def process_file(
    input_path: Path,
    output_path: Path,
    column: Optional[str] = None,
    *,
    model_dir: Optional[Path] = None,
    conflict_model_dir: Optional[Path] = None,
    train_classifier_first: bool = False,
    train_conflict_first: bool = False,
) -> tuple[pd.DataFrame, Path]:
    nlp = load_nlp()
    df = read_excel(input_path)
    root = Path(__file__).resolve().parent
    resolved_model_dir = model_dir or (root / "models" / "requirement_securebert")
    resolved_conflict_dir = conflict_model_dir or (
        root / "models" / "requirement_conflict_securebert"
    )

    classifier = None
    try:
        from classify_requirements import (
            detect_classification_column,
            load_classifier,
            train_classifier,
        )

        cls_col = detect_classification_column(df)

        if train_classifier_first and cls_col is not None:
            print("Training SecureBERT classifier on labeled rows...", file=sys.stderr)
            result = train_classifier(input_path, resolved_model_dir)
            print(
                f"Classifier trained (val F1={result.val_f1_macro:.3f}) -> {result.model_dir}",
                file=sys.stderr,
            )

        classifier = load_classifier(resolved_model_dir)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}", file=sys.stderr)
        print(
            "Proceeding without classification. Run: python train_classifier.py",
            file=sys.stderr,
        )
    except ImportError as exc:
        print(
            f"Warning: classification dependencies missing ({exc}). "
            "Install ML packages from requirements.txt.",
            file=sys.stderr,
        )

    conflict_detector = None
    requirement_records = None
    conflict_predictions: dict = {}
    conflict_col = None
    try:
        from detect_conflicts import (
            detect_conflict_column,
            detect_requirement_id_column,
            load_conflict_detector,
            load_requirement_records,
            train_conflict_detector,
        )

        conflict_col = detect_conflict_column(df)
        id_col = detect_requirement_id_column(df)

        if train_conflict_first and conflict_col is not None:
            print(
                "Training SecureBERT conflict detector on labeled pairs...",
                file=sys.stderr,
            )
            c_result = train_conflict_detector(
                input_path,
                resolved_conflict_dir,
                encoder_dir=resolved_model_dir,
            )
            print(
                f"Conflict detector trained (F1={c_result.train_f1:.3f}, "
                f"threshold={c_result.threshold:.2f}) -> {c_result.model_dir}",
                file=sys.stderr,
            )

        if id_col is not None:
            requirement_records, _, _, _ = load_requirement_records(input_path)
            conflict_detector = load_conflict_detector(
                resolved_conflict_dir, encoder_dir=resolved_model_dir
            )
            conflict_predictions = conflict_detector.predict_pairs(requirement_records)
    except FileNotFoundError as exc:
        if train_conflict_first or conflict_col is not None:
            print(f"Warning: {exc}", file=sys.stderr)
            print(
                "Proceeding without conflict detection. Run: python train_conflict_detector.py",
                file=sys.stderr,
            )
    except ImportError as exc:
        print(
            f"Warning: conflict detection dependencies missing ({exc}).",
            file=sys.stderr,
        )
    except ValueError as exc:
        print(f"Warning: {exc}", file=sys.stderr)

    req_col = column or detect_requirement_column(df)
    if req_col not in df.columns:
        raise ValueError(
            f"Column '{req_col}' not found. Available: {list(df.columns)}"
        )

    if _avg_text_length(df[req_col]) < 15:
        print(
            f"Warning: column '{req_col}' looks short (IDs?). "
            f"Use -c with the text column. Available: {list(df.columns)}",
            file=sys.stderr,
        )

    id_col_name = None
    conflict_col_name = None
    try:
        from detect_conflicts import detect_conflict_column, detect_requirement_id_column

        id_col_name = detect_requirement_id_column(df)
        conflict_col_name = detect_conflict_column(df)
    except ImportError:
        pass

    rows = []
    auto_id = 1
    for _, row in df.iterrows():
        original = row[req_col]
        if pd.isna(original) or not str(original).strip():
            continue

        req_id = ""
        if id_col_name and id_col_name in df.columns:
            raw_id = row[id_col_name]
            if pd.isna(raw_id) or not str(raw_id).strip():
                req_id = f"REQ-AUTO-{auto_id:04d}"
                auto_id += 1
            else:
                req_id = str(raw_id).strip()

        existing_conflict = ""
        if conflict_col_name and conflict_col_name in df.columns:
            raw_conflict = row[conflict_col_name]
            if not (pd.isna(raw_conflict) or not str(raw_conflict).strip()):
                existing_conflict = str(raw_conflict).strip()

        cleaned = clean_text(original)
        segments = segment_requirement(str(original), nlp)
        classification = ""
        if classifier is not None:
            classification = classifier.predict_one(str(original).strip())

        predicted_conflict = ""
        conflict_confidence = ""
        if req_id and conflict_predictions:
            hit = conflict_predictions.get(req_id)
            if hit is not None:
                predicted_conflict = hit.predicted_conflict_with
                conflict_confidence = f"{hit.confidence:.3f}"

        rows.append(
            {
                "requirement_id": req_id,
                "requirement": str(original).strip(),
                "clean": cleaned,
                "segmentation": format_segmentation(segments),
                "classification": classification,
                "existing_conflict": existing_conflict,
                "predicted_conflict": predicted_conflict,
                "conflict_confidence": conflict_confidence,
            }
        )

    out_df = pd.DataFrame(
        rows,
        columns=[
            "requirement_id",
            "requirement",
            "clean",
            "segmentation",
            "classification",
            "existing_conflict",
            "predicted_conflict",
            "conflict_confidence",
        ],
    )
    actual_output_path = output_path
    try:
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")
        out_df.to_csv(fallback, index=False, encoding="utf-8-sig")
        actual_output_path = fallback
        print(
            f"Warning: '{output_path.name}' is locked. Wrote output to '{fallback.name}' instead.",
            file=sys.stderr,
        )
    return out_df, actual_output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, segment, and classify requirements from Excel to CSV."
    )
    parser.add_argument("input", type=Path, help="Input Excel file (.xlsx or .xls)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input>_processed.csv)",
    )
    parser.add_argument(
        "-c",
        "--column",
        type=str,
        default=None,
        help="Name of the column containing requirements",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to fine-tuned SecureBERT classifier directory",
    )
    parser.add_argument(
        "--train-classifier",
        action="store_true",
        help="Retrain SecureBERT on labeled Classification column before processing",
    )
    parser.add_argument(
        "--conflict-model-dir",
        type=Path,
        default=None,
        help="Path to trained SecureBERT conflict detector directory",
    )
    parser.add_argument(
        "--train-conflict-detector",
        action="store_true",
        help="Retrain conflict detector on Existing conflict column before processing",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_processed.csv"
    )

    result, actual_output_path = process_file(
        input_path,
        output_path,
        args.column,
        model_dir=args.model_dir,
        conflict_model_dir=args.conflict_model_dir,
        train_classifier_first=args.train_classifier,
        train_conflict_first=args.train_conflict_detector,
    )
    print(f"Processed {len(result)} requirements -> {actual_output_path}")


if __name__ == "__main__":
    main()