# Echometric – Lip Sync Using Deep Learning

Echometric is a deep learning-based lip reading system that predicts spoken words by analyzing lip movements from video input. The project uses a LipNet-inspired architecture built with TensorFlow and provides a Streamlit interface for testing and demonstrations.

## Features
- Lip reading from video input
- Streamlit-based web interface
- TensorFlow/Keras deep learning model
- Video preprocessing utilities
- TensorFlow Lite optimization support

## Project Structure

Lip
│
├── app/
│   ├── streamlitapp.py
│   ├── modelutil.py
│   └── utils.py
│
├── models/
├── data_loaders.py
├── optimize_model.py
├── test_optimized_model.py
└── README.md

## Technologies Used
- Python
- TensorFlow
- Streamlit
- OpenCV
- NumPy
- Pandas

## Run the Project

1. Clone the repository

git clone https://github.com/anaghapnithyandala/Echometric-Lip-Sync-Using-Deep-Learning.git

2. Create virtual environment

python -m venv .venv

3. Activate environment

Windows:
.venv\Scripts\activate

4. Install dependencies

pip install tensorflow streamlit opencv-python imageio numpy pandas

5. Run the application

streamlit run app/streamlitapp.py

## Author
Anagha P N