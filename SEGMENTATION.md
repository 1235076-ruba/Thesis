# Automatic requirement segmentation

## Approach

Segmentation is **automatic** and driven by **spaCy linguistic structure**, not row-specific keyword rules.

Pipeline for each requirement:

1. **Clean** text (lowercase, normalize whitespace/hyphens).
2. **Parse** with spaCy (`en_core_web_sm`).
3. **Detect split candidates** using grammar only:
   - multiple sentences (`doc.sents`);
   - coordinated **verbs** (`conj` + `VERB` in the dependency tree);
   - matrix verb + adverbial clause with coordinated verb (e.g. send email when … and update inventory);
   - coordination with **different subjects** (e.g. passwords … and data …).
4. **Reject nominal-only coordination** (nouns/adjectives joined by *and/or* — e.g. “PDF and CSV”, “beautiful and modern”, step lists).
5. **Validate** every segment: must contain an obligation modal (`shall`, `must`, `can`, …) and a verb; fragments like “try again” are rejected.
6. **Propagate shared trailing constraints** when the last segment ends with `within …` / `without …` and earlier segments share the same subject prefix.

If no valid split is found, the requirement stays **atomic** (empty `segmentation` column).

## What generalizes to new requirements

Works well when new text follows similar grammar:

- `The system shall do A and shall do B`
- `Users can approve or reject items`
- `X must happen. Y must happen.` (two sentences)
- `Data must be encrypted and traffic must use TLS` (different subjects)

## Limitations

- Depends on spaCy parse quality (`en_core_web_sm` is small; typos or unusual wording may parse wrong).
- Does not “understand” domain meaning — only **syntax + validation**.
- Very novel patterns may stay atomic or produce invalid splits (then validation rejects and keeps atomic).

## Optional upgrade

For higher accuracy on diverse corpora, use `en_core_web_trf` or a trained classifier on labeled requirements.
