# Echometric Lip Reading (LipNet)

A Streamlit-based app for lip reading research and demos, built around a TensorFlow/Keras implementation of LipNet with an optional TensorFlow Lite (CPU) inference path.

## Features
- Upload a video or pick sample videos to run predictions
- Toggle preprocessing (denoise, contrast)
- Compare TensorFlow (Keras) vs TensorFlow Lite CPU inference
- Batch benchmarking and CSV export
- Utilities to generate lips-only demo videos

## Repository structure (key paths)
- `app/streamlitapp.py`: Streamlit UI and inference logic
- `app/modelutil.py`: Builds the Keras model and loads weights
- `app/utils.py`: Video loading, tokenization utilities
- `models/`: Model artifacts
  - `ckpt96/` and `ckpt50/`: Trained Keras checkpoints (prefix `checkpoint`)
  - `lipnet_dynamic.tflite`: TFLite model (generated)
- `optimize_model.py`: Convert Keras model to TensorFlow Lite
- `test_optimized_model.py`: Small benchmark comparing Keras vs TFLite

## Prerequisites
- Python 3.9–3.11 (Windows/macOS/Linux). For Windows, use a Python version compatible with TensorFlow 2.x.
- FFmpeg available on PATH (required for preview/transcoding):
  - Windows (choco): `choco install ffmpeg -y`
  - Windows (scoop): `scoop install ffmpeg`
  - macOS (brew): `brew install ffmpeg`
  - Linux (apt): `sudo apt-get update && sudo apt-get install -y ffmpeg`

## Create and activate a virtual environment
```bash
# From the project root
python -m venv .venv

# Windows PowerShell
. .venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

## Install dependencies
If you have a GPU and a compatible CUDA setup, install the GPU build of TensorFlow instead of the CPU build.

```bash
pip install --upgrade pip

# Core runtime
pip install tensorflow==2.*

# App dependencies
pip install streamlit opencv-python imageio numpy pandas
```

If OpenCV wheels fail on your platform, you may use `opencv-python-headless` as a fallback.

## Model weights
The Keras model expects a checkpoint prefix path (without extension) when loading weights. This repository includes two trained checkpoints:

- High-accuracy (recommended): `models/ckpt96/checkpoint`
- Alternative: `models/ckpt50/models/checkpoint` (note the extra `models/` subfolder)

You can configure the weights in two ways:

- Via environment variable before launching Streamlit:
```bash
# Example: use the ckpt96 weights
$env:MODEL_WEIGHTS_PATH = "models/ckpt96/checkpoint"   # Windows PowerShell
# export MODEL_WEIGHTS_PATH=models/ckpt96/checkpoint     # macOS/Linux
```

- Or directly inside the app sidebar under "Keras weights path".

If you see a "Model failed to load" error, double-check the path points to the checkpoint prefix (no file extension).

## Run the Streamlit app
From the project root:
```bash
streamlit run app/streamlitapp.py
```
Then open the URL shown in the terminal (typically `http://localhost:8501`).

Inside the app:
- Choose inference backend: "TensorFlow (Keras)" or "TensorFlow Lite (CPU)"
- Set "Keras weights path" (or rely on the env var)
- Optionally set "TFLite model path" (defaults to `models/lipnet_dynamic.tflite`)
- Upload a `.mp4`/`.mpg` video or pick a built-in sample
- Use batch tools and preprocessing toggles as needed

## Generate a TensorFlow Lite model (optional)
If you want to use the TFLite CPU backend, generate the `.tflite` file first:
```bash
# Ensure MODEL_WEIGHTS_PATH points to a valid checkpoint
python optimize_model.py
```
This will create `models/lipnet_dynamic.tflite` and print the source weights path for verification.

## Test and benchmark the optimized model (optional)
```bash
python test_optimized_model.py
```
The script will:
- Load a few `.mpg` samples from `../data/s1` if present
- Compare average/median inference times for Keras vs TFLite

## Data notes
- The original LipNet pipeline uses GRID corpus alignment files. This app can run without alignments using the built-in demos or your own videos.
- For GRID-style samples, place `.mpg` files under `../data/s1` relative to `app/` (i.e., `Lip/data/s1/...`).

## Troubleshooting
- "Model failed to load": Ensure `MODEL_WEIGHTS_PATH` or the sidebar value points to the correct checkpoint prefix (e.g., `models/ckpt96/checkpoint`).
- Video preview fails: Verify FFmpeg is installed and available on PATH. The app will fall back to the original file if transcoding fails.
- TensorFlow import errors on Windows: Use a supported Python version (3.9–3.11) and ensure VC++ redistributables and compatible CPU/GPU runtimes are present.
- OpenCV issues on servers: Try `opencv-python-headless`.

## Quick start (Windows PowerShell)
```powershell
cd C:\Users\ASUSZENBOOK14\Desktop\Lip
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install tensorflow==2.* streamlit opencv-python imageio numpy pandas
$env:MODEL_WEIGHTS_PATH = "models/ckpt96/checkpoint"
streamlit run app/streamlitapp.py
```

---
Maintained as a research/demo app for lip reading experiments. Contributions and issues welcome.
