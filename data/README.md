# Data

Two things are needed that are **not** in this repository. Both take a few minutes.

## 1. The RBI documents

RBI's website reserves all rights to its content and doesn't allow copies to be redistributed, so save the pages yourself. Open each page, press **Ctrl+S**, choose **"Webpage, HTML only"**, and save into `data/raw/` with exactly this name:

| Save as | Page |
|---|---|
| `customer_liability.html` | https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=11040 |
| `digital_lending.html` | https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12848 |
| `fraud_risk.html` | https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=12702 |

The pages contain the full text of each document. RBI's PDF links often refuse direct downloads, and the HTML keeps headings and clause numbers, which the parser uses for citations. Then run `python -m grounded.ingest`.

## 2. The embedding model

[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (MIT licence), ONNX version, 133 MB:

```bash
pip install "huggingface_hub[cli]"
hf download BAAI/bge-small-en-v1.5 onnx/model.onnx vocab.txt --local-dir models/bge-small-en-v1.5
```

Tip: do this in the project's virtual environment. Upgrading `huggingface_hub` in your global Python can break other projects that pin an older version.

## What *is* in the repository

- `evals/testset.yaml`: the 65 evaluation questions, gold clause ids and required facts (short phrases only).
- `reports/`: evaluation results.
- `tests/fixtures/sample_rbi.html`: an invented page with RBI's structure, used by the unit tests.
