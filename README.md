# Requirement clean & segmentation (spaCy)

Reads requirement text from an **Excel** file (`.xlsx` or `.xls`), normalizes it, and **automatically segments** compound requirements using spaCy (dependency tree + sentence boundaries + segment validation). No row-specific keyword rules. See `SEGMENTATION.md` for details.

Writes a **CSV** with:

| Column          | Meaning |
|----------------|---------|
| `requirement`  | Original cell text |
| `clean`        | Lowercase, hyphenation fixed, extra symbols removed |
| `segmentation` | Numbered segments joined by ` \|\| `; **empty** if there is only one segment |

## Setup

Use **Python 3.10–3.12** if possible; spaCy wheels for **3.14** may be incomplete on some machines.

```powershell
cd "c:\Users\PH-User\Desktop\Project"
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```powershell
python process_requirements.py "C:\path\to\your\requirements.xlsx"
```

Optional:

- `-o output.csv` — output path (default: `<input_stem>_processed.csv`)
- `-c "Requirement text"` — exact column name (optional). If omitted, the script uses **`Requirement text`** / **`Requirement_DS`** when present, and skips ID columns like **`Requirement ID`**.
