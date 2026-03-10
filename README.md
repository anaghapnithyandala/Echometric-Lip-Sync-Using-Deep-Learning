# Echometric – Lip Sync Using Deep Learning

Echometric is a deep learning–based lip reading system that predicts spoken words by analyzing lip movements from video input. The project is inspired by the LipNet architecture and is implemented using TensorFlow and Streamlit to provide an interactive interface for testing and demonstrating lip reading models.

This system processes video frames, extracts lip region features, and predicts the spoken text using a trained neural network model.

---

## Features

* Lip reading from uploaded video clips
* Deep learning model built with TensorFlow/Keras
* Streamlit-based interactive web interface
* Video preprocessing and lip-region extraction
* Optional TensorFlow Lite CPU inference support
* Benchmark comparison between TensorFlow and TensorFlow Lite

---

## Project Structure

```
Lip
│
├── app/
│   ├── streamlitapp.py          # Streamlit web application
│   ├── modelutil.py             # Model loading and configuration
│   ├── utils.py                 # Video preprocessing utilities
│   └── animation.gif            # Demo animation
│
├── models/                      # Model checkpoints and trained weights
│
├── convert_checkpoint_to_h5.py  # Convert model checkpoints
├── data_loaders.py              # Dataset loading utilities
├── optimize_model.py            # Convert model to TensorFlow Lite
├── test_optimized_model.py      # Benchmark optimized model
│
└── README.md                    # Project documentation
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* OpenCV
* NumPy
* Pandas
* ImageIO

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/Madhu-1106/Echometric-Lip-Sync-Using-Deep-Learning.git
cd Echometric-Lip-Sync-Using-Deep-Learning
```

### 2. Create a virtual environment

```
python -m venv .venv
```

### 3. Activate the environment

Windows:

```
.venv\Scripts\activate
```

Mac/Linux:

```
source .venv/bin/activate
```

### 4. Install dependencies

```
pip install tensorflow streamlit opencv-python imageio numpy pandas
```

---

## Running the Application

Start the Streamlit app:

```
streamlit run app/streamlitapp.py
```

After running the command, open the URL shown in the terminal (usually):

```
http://localhost:8501
```

---

## Model Weights

The project supports multiple trained checkpoints stored in the `models` folder.

Example paths:

```
models/ckpt96/checkpoint
models/ckpt50/checkpoint
```

You can configure the weights using an environment variable.

Windows:

```
$env:MODEL_WEIGHTS_PATH="models/ckpt96/checkpoint"
```

Linux/Mac:

```
export MODEL_WEIGHTS_PATH=models/ckpt96/checkpoint
```

---

## TensorFlow Lite Optimization (Optional)

To convert the trained model to TensorFlow Lite:

```
python optimize_model.py
```

This will generate:

```
models/lipnet_dynamic.tflite
```

---

## Benchmark the Optimized Model

```
python test_optimized_model.py
```

This script compares performance between:

* TensorFlow (Keras) model
* TensorFlow Lite optimized model

---

## Applications

* Assistive technology for hearing-impaired individuals
* Speech recognition research
* Human–computer interaction systems
* Digital animation and virtual avatars
* Video communication enhancement

---

## Future Improvements

* Real-time webcam lip reading
* Support for larger datasets
* Improved accuracy using transformer architectures
* Integration with speech synthesis systems

---

## Author

Vijay Raj M

Final Year AI & ML Student
Project: Echometric – Lip Sync Using Deep Learning

---

## License

This project is for educational and research purposes.
