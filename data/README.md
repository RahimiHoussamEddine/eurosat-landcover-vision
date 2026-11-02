# Data

`make fetch` downloads EuroSAT RGB v2 from the authors' Zenodo record. The
RGB archive contains 27,000 64×64 Sentinel-2 JPEG patches across 10 land-cover
classes. Its official record reports an archive size of 94.7 MB and an MD5
checksum; the acquisition script verifies the source checksum before it is
used and records a SHA-256 digest in the build manifest.

The upstream EuroSAT authors identify the dataset as MIT-licensed and request
consideration of the Copernicus Sentinel data terms. See [NOTICE.md](../NOTICE.md).
Raw archive and extracted image files are ignored by Git. This repository
commits the compact trained model and aggregate evaluation artifacts only.

Run `make build` after fetching the data to audit image integrity, create
stratified duplicate-group-aware development/validation/test splits, train a
small residual CNN and HOG/linear-SVM baseline, and generate evaluation
artifacts.
