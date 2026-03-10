# Echometric – Lip Sync Using Deep Learning

Echometric is a deep learning–based lip reading system that predicts spoken words by analyzing lip movements from video. The project implements a LipNet-based architecture using TensorFlow and provides a Streamlit web interface for real-time lip-reading experiments and demonstrations.

## Features
- Lip reading from uploaded video clips
- Streamlit-based interactive interface
- TensorFlow/Keras deep learning model
- Optional TensorFlow Lite CPU inference
- Benchmark comparison between Keras and TFLite
- Utilities for preprocessing and lip-region extraction

## Project Structure

Lip
│
├── app/
│   ├── streamlitapp.py
│   ├── modelutil.py
│   └── utils.py
│
├── models/
├── optimize_model.py
├── test_optimized_model.py
├── data_loaders.py
└── README.md

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- OpenCV
- NumPy
- Pandas

## How to Run the Project

1. Clone the repository

git clone https://github.com/Madhu-1106/Echometric-Lip-Sync-Using-Deep-Learning.git

2. Create virtual environment

python -m venv .venv

3. Activate environment

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

4. Install dependencies

pip install tensorflow streamlit opencv-python imageio numpy pandas

5. Run the application

streamlit run app/streamlitapp.py

## Applications

- Assistive technology for hearing-impaired users
- Speech recognition research
- Human-computer interaction
- Digital animation and virtual avatars

## Author

Madhu M S
