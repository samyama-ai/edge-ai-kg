"""The fictional archetypes the fleet is generated from.

Every name here is invented. Vendor, SoC, board and runtime names are
deliberately not real products, so no figure derived from them can be mistaken
for a claim about one -- the only real data in this KG is the ONNX operator
catalog (`etl/onnx_catalog.py`).

These are plain tables, not generated: `etl/generate.py` draws from them with
its seeded `random.Random`, so they are the fixed half of the determinism
guarantee and this module holds no randomness at all.

Split out of `etl/generate.py`, which had grown past the 500 lines the review
harness reads before it stops.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Fictional vendors. Not real companies.
# --------------------------------------------------------------------------
VENDORS = [
    ("Corvid Silicon", "IN"), ("Nimbus Micro", "US"), ("Tessera Labs", "DE"),
    ("Akshara Semi", "IN"), ("Halcyon Devices", "JP"), ("Vermilion Systems", "US"),
    ("Kestrel Embedded", "UK"), ("Suvarna Microsystems", "IN"),
]

# Accelerator archetypes: (kind, op categories it can run, opset ceiling,
#   int8 GOPS range, sram KB range, relative energy per op)
ACCEL_ARCHETYPES = [
    ("MCU-CPU", {"elementwise", "activation", "shape", "reduction", "tensor",
                 "matmul", "convolution", "spatial", "normalization", "recurrent",
                 "quantization", "signal", "attention", "stochastic",
                 "loss"}, 99,
     (0.5, 3.0), (64, 512), 1.00),
    ("DSP", {"elementwise", "activation", "signal", "reduction", "matmul",
             "convolution", "spatial", "shape"}, 17, (8.0, 40.0), (256, 2048), 0.42),
    ("NPU-Lite", {"convolution", "matmul", "activation", "spatial", "normalization",
                  "elementwise", "quantization"}, 13, (30.0, 120.0), (512, 4096), 0.16),
    ("NPU-Pro", {"convolution", "matmul", "activation", "spatial", "normalization",
                 "elementwise", "quantization", "reduction", "attention", "shape"}, 19,
     (150.0, 900.0), (2048, 16384), 0.11),
    ("GPU-Embedded", {"convolution", "matmul", "activation", "spatial", "normalization",
                      "elementwise", "quantization", "reduction", "attention", "shape",
                      "recurrent", "tensor"}, 21, (400.0, 2400.0), (4096, 32768), 0.30),
]

RUNTIMES = [
    # (name, version, model format, accelerator kinds it can target)
    ("TFLite Micro", "1.4", "tflite", {"MCU-CPU", "DSP", "NPU-Lite"}),
    ("ONNX Runtime", "1.20", "onnx", {"MCU-CPU", "DSP", "NPU-Lite", "NPU-Pro", "GPU-Embedded"}),
    ("ExecuTorch", "0.5", "pte", {"MCU-CPU", "NPU-Lite", "NPU-Pro", "GPU-Embedded"}),
    ("TVM microTVM", "0.17", "tar", {"MCU-CPU", "DSP", "NPU-Lite", "NPU-Pro"}),
    ("CMSIS-NN", "6.0", "c-array", {"MCU-CPU"}),
    ("Vendor SDK", "3.1", "blob", {"DSP", "NPU-Lite", "NPU-Pro"}),
    ("OpenVINO", "2025.1", "ir", {"MCU-CPU", "GPU-Embedded", "NPU-Pro"}),
]

# Model architecture families -> operator categories they draw from, and a
# rough MAC budget. Task assignment comes from CLINICAL_TASKS below.
MODEL_FAMILIES = [
    ("tiny-cnn",    ["convolution", "activation", "spatial", "normalization", "elementwise", "shape"], (0.4, 12.0)),
    ("depthwise-cnn", ["convolution", "activation", "spatial", "normalization", "elementwise", "shape", "reduction"], (2.0, 45.0)),
    ("resnet-1d",   ["convolution", "activation", "normalization", "elementwise", "reduction", "shape"], (5.0, 90.0)),
    ("lstm-seq",    ["recurrent", "matmul", "activation", "elementwise", "shape"], (1.0, 30.0)),
    ("gru-seq",     ["recurrent", "matmul", "activation", "elementwise", "shape"], (0.8, 22.0)),
    ("tiny-transformer", ["attention", "matmul", "normalization", "activation", "elementwise", "shape", "reduction"], (12.0, 240.0)),
    ("spectro-cnn", ["signal", "convolution", "activation", "spatial", "normalization", "elementwise"], (3.0, 60.0)),
    ("mlp-features", ["matmul", "activation", "elementwise", "shape"], (0.1, 4.0)),
]

PRECISIONS = [
    # (name, size multiplier vs fp32, throughput multiplier, accuracy delta)
    ("fp32", 1.00, 1.00, 0.000),
    ("fp16", 0.50, 1.70, -0.002),
    ("int8", 0.25, 3.20, -0.011),
    ("int4", 0.14, 4.80, -0.037),
]

SENSORS = [
    ("ECG 3-lead", "ecg", 500, 3, 16), ("ECG 12-lead", "ecg", 1000, 12, 16),
    ("PPG wrist", "ppg", 128, 2, 14), ("PPG fingertip", "ppg", 256, 1, 14),
    ("EEG 8-channel", "eeg", 256, 8, 24), ("EEG 32-channel", "eeg", 512, 32, 24),
    ("EMG surface", "emg", 2000, 4, 16), ("IMU 6-axis", "imu", 200, 6, 16),
    ("IMU 9-axis", "imu", 400, 9, 16), ("Contact mic", "audio", 16000, 1, 16),
    ("Chest mic array", "audio", 8000, 4, 16), ("Thermistor array", "temp", 4, 8, 12),
    ("SpO2 optical", "spo2", 64, 2, 14), ("Bio-impedance", "bioz", 100, 4, 16),
]

# Signal-processing stages. `ops` are the ONNX operator categories the stage
# maps onto once it is lowered into the graph.
SIGNAL_STAGES = [
    ("DC removal", "filter", 20, 0.4, ["elementwise", "reduction"]),
    ("Bandpass 0.5-40Hz", "filter", 200, 3.1, ["elementwise", "convolution"]),
    ("Notch 50Hz", "filter", 200, 1.8, ["elementwise", "convolution"]),
    ("Baseline wander removal", "filter", 400, 4.2, ["elementwise", "convolution", "reduction"]),
    ("Resample", "transform", 100, 1.2, ["spatial", "shape"]),
    ("Windowing (Hann)", "transform", 64, 0.3, ["signal", "elementwise"]),
    ("STFT", "transform", 256, 12.5, ["signal", "matmul"]),
    ("Mel filterbank", "transform", 256, 6.0, ["signal", "matmul"]),
    ("Wavelet decomposition", "transform", 512, 18.0, ["convolution", "elementwise"]),
    ("R-peak detection", "feature", 800, 2.6, ["reduction", "shape", "elementwise"]),
    ("HRV features", "feature", 30000, 0.9, ["reduction", "elementwise"]),
    ("Spectral entropy", "feature", 512, 3.4, ["reduction", "elementwise", "signal"]),
    ("Z-score normalize", "transform", 100, 0.5, ["normalization", "reduction"]),
    ("Artifact rejection", "filter", 1000, 5.5, ["reduction", "elementwise", "shape"]),
    ("Downsample decimate", "transform", 100, 0.7, ["spatial", "shape"]),
    ("Envelope extraction", "feature", 300, 2.0, ["elementwise", "convolution"]),
]

# (task, category, sensor modalities, latency budget ms, min sensitivity)
CLINICAL_TASKS = [
    ("Atrial fibrillation detection", "cardiac", ["ecg", "ppg"], 1000, 0.95),
    ("Ventricular arrhythmia alarm", "cardiac", ["ecg"], 250, 0.98),
    ("QRS morphology classification", "cardiac", ["ecg"], 500, 0.92),
    ("Heart-rate variability scoring", "cardiac", ["ecg", "ppg"], 5000, 0.85),
    ("Blood-pressure surrogate estimation", "cardiac", ["ppg", "bioz"], 2000, 0.80),
    ("Seizure onset detection", "neuro", ["eeg"], 300, 0.97),
    ("Sleep-stage classification", "neuro", ["eeg", "imu"], 30000, 0.86),
    ("Cognitive-load estimation", "neuro", ["eeg", "ppg"], 4000, 0.78),
    ("Tremor quantification", "neuro", ["imu", "emg"], 1000, 0.88),
    ("Gait-anomaly detection", "mobility", ["imu"], 2000, 0.84),
    ("Fall detection", "mobility", ["imu"], 200, 0.96),
    ("Freezing-of-gait prediction", "mobility", ["imu", "emg"], 500, 0.90),
    ("Cough event classification", "respiratory", ["audio"], 1000, 0.89),
    ("Wheeze detection", "respiratory", ["audio"], 1500, 0.91),
    ("Respiratory-rate estimation", "respiratory", ["audio", "bioz", "imu"], 5000, 0.87),
    ("Apnea-event detection", "respiratory", ["spo2", "audio"], 10000, 0.94),
    ("Hypoxemia early warning", "respiratory", ["spo2", "ppg"], 3000, 0.95),
    ("Muscle-fatigue estimation", "musculoskeletal", ["emg"], 2000, 0.82),
]

DATASETS = [
    ("MIT-BIH Arrhythmia", "PhysioNet", 47, 24, "ODC-BY 1.0"),
    ("PTB-XL ECG", "PhysioNet", 18869, 5300, "CC-BY 4.0"),
    ("CinC Challenge 2017", "PhysioNet", 8528, 240, "ODC-BY 1.0"),
    ("CHB-MIT Scalp EEG", "PhysioNet", 24, 980, "ODC-BY 1.0"),
    ("Sleep-EDF Expanded", "PhysioNet", 197, 3800, "ODC-BY 1.0"),
    ("TUH EEG Corpus", "Temple University", 14000, 27000, "Custom research"),
    ("MobiAct", "TEI of Crete", 67, 92, "Research use"),
    ("UCI HAR", "UCI ML Repository", 30, 25, "CC-BY 4.0"),
    ("Daphnet FoG", "UCI ML Repository", 10, 35, "CC-BY 4.0"),
    ("ICBHI Respiratory", "ICBHI Challenge", 126, 5, "Research use"),
    ("Coswara", "IISc Bangalore", 2000, 60, "CC-BY 4.0"),
    ("BIDMC PPG and Respiration", "PhysioNet", 53, 8, "ODC-BY 1.0"),
]

CERTIFICATIONS = [
    ("IEC 62304 Class A", "IEC", "A"), ("IEC 62304 Class B", "IEC", "B"),
    ("IEC 62304 Class C", "IEC", "C"), ("ISO 13485", "ISO", "QMS"),
    ("FDA 510(k) Class II", "FDA", "II"), ("EU MDR Class IIa", "EU", "IIa"),
]

FORM_FACTORS = ["wearable-band", "patch", "chest-module", "handheld",
                "bedside-module", "implant-adjacent", "m.2-module", "som"]

