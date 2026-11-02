# Experiment protocol

## Question

Can a compact image model classify EuroSAT RGB land-cover patches accurately,
which classes account for most errors, and how much does mild image corruption
change its performance?

## Data and split

Use the complete official EuroSAT RGB v2 archive. Audit image count, dimensions,
file decoding, class frequencies, and exact duplicate content. Build a seeded
stratified group split by combining 3 of 20 outer folds for test, then 3 of 17
development folds for validation. This targets a 70/15/15 train/validation/test
split; reported row counts are authoritative because group fold sizes can vary.
Exact duplicate images share a split group. Do not make geographic transfer
claims because the RGB archive does not provide an analysis-ready spatial
holdout.

## Models

Train a compact residual CNN from random initialization with geometric
augmentation that preserves land-cover semantics. Use validation macro-F1 for
early stopping. Compare with a HOG feature extractor and linear SVM trained on
the identical training images. Do not use pretrained ImageNet weights.

## Evaluation

Report test accuracy, balanced accuracy, macro-F1, per-class precision/recall,
top-3 accuracy, multiclass Brier score, and a class-confusion matrix. Estimate
95% intervals for accuracy and macro-F1 using a seeded, class-stratified
bootstrap of the locked test set. Evaluate the CNN again with mild Gaussian
blur and brightness shift. Use the same clean test rows in each condition.

## Limits

This is a benchmark image classifier, not an operational land-cover map. The
dataset is geographically sampled and the random split does not measure
performance on new regions, dates, sensors, or spatially separated tiles.
Labels represent the benchmark's 10 broad classes; several are visually
similar and class errors are not interchangeable in an application. Stress
tests are controlled corruptions, not a comprehensive satellite-domain shift
study. Prediction scores are model confidence, not calibrated probabilities
unless calibration validation demonstrates otherwise.
