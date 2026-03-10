import os
import time
import numpy as np
import tensorflow as tf

from app.modelutil import load_model
from app.utils import load_video, num_to_char


def load_tflite_interpreter(tflite_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_keras(model: tf.keras.Model, video_tensor: tf.Tensor) -> str:
    yhat = model.predict(tf.expand_dims(video_tensor, axis=0), verbose=0)
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    return tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')


def predict_tflite(interpreter: tf.lite.Interpreter, video_tensor: tf.Tensor) -> str:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ensure dtype and shape
    input_data = tf.expand_dims(video_tensor, axis=0)
    input_data = tf.cast(input_data, input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data.numpy())
    interpreter.invoke()
    yhat = interpreter.get_tensor(output_details[0]['index'])
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    return tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')


def ensure_75_frames(frames_tensor: tf.Tensor) -> tf.Tensor:
    num_frames = tf.shape(frames_tensor)[0]
    target_frames = 75
    if int(num_frames) >= target_frames:
        return frames_tensor[:target_frames]
    pad_len = target_frames - int(num_frames)
    pad_tensor = tf.zeros((pad_len, frames_tensor.shape[1], frames_tensor.shape[2], frames_tensor.shape[3]), dtype=frames_tensor.dtype)
    return tf.concat([frames_tensor, pad_tensor], axis=0)


def benchmark(video_paths, runs: int = 5):
    model = load_model()

    tflite_path = os.path.join('models', 'lipnet_dynamic.tflite')
    if not os.path.exists(tflite_path):
        raise FileNotFoundError('Run optimize_model.py first to generate models/lipnet_dynamic.tflite')
    interpreter = load_tflite_interpreter(tflite_path)

    for path in video_paths:
        video_tensor = load_video(path)
        video_tensor = ensure_75_frames(video_tensor)

        # Warmup
        _ = predict_keras(model, video_tensor)
        _ = predict_tflite(interpreter, video_tensor)

        # Timing
        keras_times = []
        tflite_times = []
        for _ in range(runs):
            t0 = time.time(); _ = predict_keras(model, video_tensor); keras_times.append(time.time() - t0)
            t0 = time.time(); _ = predict_tflite(interpreter, video_tensor); tflite_times.append(time.time() - t0)

        print(f'Video: {path}')
        print(f'Keras avg: {np.mean(keras_times):.4f}s, med: {np.median(keras_times):.4f}s')
        print(f'TFLite avg: {np.mean(tflite_times):.4f}s, med: {np.median(tflite_times):.4f}s')


if __name__ == '__main__':
    # Example usage: test a couple of known sample files under ../data/s1
    sample_dir = os.path.join('..', 'data', 's1')
    if os.path.isdir(sample_dir):
        candidates = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith('.mpg')]
        candidates = candidates[:2]
        if candidates:
            benchmark(candidates, runs=3)
        else:
            print('No sample .mpg files found under ../data/s1')
    else:
        print('Sample directory ../data/s1 not found')


