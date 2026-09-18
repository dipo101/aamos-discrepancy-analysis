# Raw data

Place the three timestamped AAMOS-00 exports here (they are git-ignored):

- `anonym_aamos00_patient_info.csv`
- `anonym_aamos00_dailyquestionnaire_dt.csv`
- `anonym_aamos00_smartinhaler_dt.csv`

and optionally `aamos00-end-final-freetext.csv` for the feedback analyses.

`MANIFEST.sha256` **is** committed. Every loader verifies the files against it
before reading and refuses to run on a mismatch. After intentionally replacing
the data, regenerate and commit it:

```bash
python -m aamos_concordance --write-manifest
```

Verify by hand with `python -m aamos_concordance --verify` or, inside this
folder, `shasum -a 256 -c MANIFEST.sha256`.

Alternatively set `AAMOS_DATA_DIR` to a folder elsewhere on disk.
