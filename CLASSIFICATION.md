# Requirement classification (SecureBERT)

Automated **clear / unclear / incomplete** labeling using **SecureBERT** (not fixed rules).

## How training works

| Dataset size | Method |
|--------------|--------|
| **&lt; 100 labeled rows** (your 31-row set) | SecureBERT **embeddings** + balanced logistic classifier (recommended) |
| **≥ 100 labeled rows** | Full SecureBERT fine-tune (optional: `--finetune`) |

With only ~31 examples, full fine-tuning cannot learn (all classes were ~33% probability → everything predicted as `clear`). The embedding approach uses SecureBERT to understand text, then trains a small classifier on top — appropriate for small datasets.

## Labels

| Label | Meaning |
|-------|---------|
| `clear` | Specific, measurable, unambiguous |
| `unclear` | Vague or subjective (e.g. "load quickly") |
| `incomplete` | Missing key details (when, how, what type, etc.) |

## Setup

```powershell
cd "c:\Users\PH-User\Desktop\Project"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Default base model: `ehsanaghaei/SecureBERT`  
Override: `$env:SECUREBERT_MODEL = "cisco-ai/SecureBERT2.0-base"`

## Train (uses `Classification` column in Excel)

Only rows with a non-empty **Classification** value are used.

```powershell
python train_classifier.py Requirement_DS.xlsx
```

Model is saved to `models/requirement_securebert/`.

## Process requirements (clean + segment + classify)

```powershell
python process_requirements.py Requirement_DS.xlsx -o Requirement_DS_processed.csv
```

Output columns:

- `requirement` — original text  
- `clean` — normalized text  
- `segmentation` — atomic segments (if any)  
- `classification` — **predicted** label (for new rows without labels too)

Train and process in one step:

```powershell
python process_requirements.py Requirement_DS.xlsx --train-classifier
```

## New requirements without labels

Add rows to `Requirement_DS.xlsx` with **Requirement text** only (leave **Classification** empty).  
Run `process_requirements.py` — the model predicts `classification` for every row.

Retrain when you add many new labeled examples:

```powershell
python train_classifier.py Requirement_DS.xlsx --epochs 10
```
