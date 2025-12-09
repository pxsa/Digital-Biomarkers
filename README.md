# Anxiety Detection Using Digital Biomarkers and Machine Learning


## Datasets

- [A Wearable Exam Stress Dataset for Predicting Cognitive Performance in Real-World Settings](https://physionet.org/content/wearable-exam-stress/1.0.0/)
- [A Wearable Exam 2025](https://www.mdpi.com/1424-8220/25/18/5628)

## Preprocessing

Here's a clear, practical guide to preprocess time-series signals and prepare them for an ML model.

### Quick overview

- Clean the signal (remove noise/artifacts).
- Make examples (windows/segments) of uniform length.
- Extract features (or use raw windows) that ML can learn from.
- Correctly label, balance and split data.
- Save datasets in an efficient format and use reproducible preprocessing.


### Cleaning / denoising

- `Bandpass filter` to remove baseline wander & high-freq noise 
    - ECG: 0.5-40 Hz or 0.05-100 Hz depending.
- Median baseline removal: subtract a long median(200-600 ml) to remove wandering baseline if needed.
- Notch filter at 50/60 Hz to remove mains interference (if present).
- Artifact removal
    - detect large amplitude spikes or saturations and either remove the segment or interpolate.

- Resample all signals to a common `fs` so features/windows line up.

### Windowing / segmentation

Two main strategies:
- Beat-level (centered on detected R-peaks)
    - good for heartbeat classification(PVC vs normal).
- Fixed-length sliding windows(with overlap, e.g., 5s windows with 50% overlap)
    - good for rhythm classification or arrhythmia detection.


### Normalization & scaling

- Per-record normalization
    - subtract median or mean, divide by std(helps if sensors/records vary).
- Per-window normalization
        - when absolute amplitude is not meaningful, scale each window (z-score or min-max).
- Global scaling: fit scaler on training set only, apply to validation/test.

### Feature Extraction

- Feed raw windows to a model 
    - CNN
    - 1D-CNN
    - RNN
    - Transformer
- Compute `handcrafted features` for classical models
    - SVM
    - Random Forest
    - XGBOOST

#### `Useful time-domain features:`

- mean, std, variance, RMS, median, IQR
- peak-to-peak amplitude, number of zero-crossings
- skewness, kurtosis
- signal energy, entropy(Shannon)
- RR interval, HRV metrics (SDNN, RMSSD) if you detect beats

#### `Useful frequency-domain features:`

- Power spectral density
- Spectral centroid, spectral entropy, dominant frequency

#### Useful morphological / shape features (ECG-specific):

- QRS duration, R-peak amplitude, R to R variablity, normalized area under QURS, slopes on rising/falling edges


### Labeling & alignment

- Ensure labels align with windows:
    - For beat-level labels, center windows on annotated beats.
    - For rhythm labels(pe recording), assign the same label to all windows from that recording or to specific time ranges.
- If labels are sparse (event annotations), generate windows around event and mark others as negative.

### Data augmentation (to reduce overfitting)


### Class imbalance handling

### Splitting strategy

 - Subject-wise split: always prefer splitting by subject/patient so the model generalizes to unseen subjects.
 - Maintain calss balance across splits.
 - Use cross-validation at subject level when possible.

 ## Evaluation metrics

 - Classification: accuracy, F1(macro/micro), precision, recall, ROC-AUC (per-class). For imbalanced data prefer F1/precision/recall.

 - Beat detection: sensitivity/PPV (true positives relative to annotated peaks) with tolerance window