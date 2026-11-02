# EuroSAT Land-Cover Vision

An end-to-end computer vision benchmark for classifying 64×64 RGB Sentinel-2
land-cover patches. The project audits the full EuroSAT RGB v2 archive, trains
a compact residual CNN from scratch, compares it with HOG + linear SVM, and
reports class-level errors and sensitivity to mild image corruption.

## Results

Run `make fetch`, then `make build` to reproduce the benchmark. The generated
report and figures are in [`reports/RESULTS.md`](reports/RESULTS.md) and
[`reports/figures/`](reports/figures/). The test set is held out from training
and model selection; exact duplicate images are grouped into one split.

## Run locally

Python 3.11–3.12 and a CPU are sufficient. The commands use a project-local
`.venv` when it exists. Dataset acquisition downloads the official 94.7 MB
archive and verifies its published checksums.

```bash
python3 -m venv .venv
make install-dev
make fetch
make build
make verify
make test
make quality
make app
```

The Streamlit explorer is available at `http://localhost:8501`. The optional
FastAPI service runs with `make api` at `http://localhost:8000`; its interactive
API page is at `/docs`. The API accepts JPEG or PNG images at `/predict` and
returns the top three classes with uncalibrated softmax confidence scores.

## Docker

After generating the model artifacts, build and run the interactive explorer:

```bash
docker build -t eurosat-landcover-vision .
docker run --rm -p 8501:8501 eurosat-landcover-vision
```

## Data and method

EuroSAT RGB v2 contains 27,000 images in 10 classes. The acquisition script
checks the source archive's MD5 and SHA-256 before safely extracting it. The
analysis audits dimensions, decodability, per-class counts, and exact byte
duplicates. It creates a seeded, duplicate-group-aware 70/15/15 train,
validation, and test split. The CNN uses random initialization and validation
macro F1 for early stopping; HOG + linear SVM uses the same training split.
No pretrained ImageNet weights are used.

The evaluation includes accuracy, balanced accuracy, macro F1, top-3 accuracy,
per-class precision and recall, multiclass Brier score, bootstrap confidence
intervals, a confusion matrix, and two controlled corruption checks. See the
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for details.

## Limits and attribution

The random split is not a geographic holdout and does not measure transfer to
new regions, dates, sensors, or spatially separated tiles. This benchmark model
is not an operational land-cover mapping product. Softmax scores are not
calibrated probabilities. Dataset terms and required attribution are recorded
in [`NOTICE.md`](NOTICE.md); raw and extracted images are excluded from Git.

Dataset sources: [EuroSAT authors' repository](https://github.com/phelber/EuroSAT),
[official EuroSAT RGB v2 archive](https://zenodo.org/records/7711810), and the
[EuroSAT dataset paper](https://arxiv.org/abs/1709.00029).
