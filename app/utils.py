import tensorflow as tf
from typing import List
import cv2
import os 
import numpy as np

# Vocabulary and pure-Python maps (no tf.constant at import time)
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
_py_char2id = {c: i + 1 for i, c in enumerate(vocab)}  # 0 reserved for OOV/blank
_py_id2char = {i + 1: c for i, c in enumerate(vocab)}

# numpy_function helpers

def _np_chars_to_ids(byte_arr: np.ndarray) -> np.ndarray:
    # byte_arr is 1-D array of bytes objects
    out = np.zeros(byte_arr.shape[0], dtype=np.int64)
    for i, b in enumerate(byte_arr):
        try:
            ch = b.decode('utf-8') if isinstance(b, (bytes, bytearray)) else str(b)
        except Exception:
            ch = ''
        out[i] = _py_char2id.get(ch, 0)
    return out


def _np_ids_to_chars(id_arr: np.ndarray) -> np.ndarray:
    # id_arr is 1-D array of ints
    out = np.empty(id_arr.shape[0], dtype=object)
    for i, val in enumerate(id_arr.tolist()):
        out[i] = _py_id2char.get(int(val), '')
    return out.astype('S')  # bytes strings for tf.string


def char_to_num(tokens: tf.Tensor) -> tf.Tensor:
    flat = tf.reshape(tokens, (-1,))
    ids = tf.numpy_function(_np_chars_to_ids, [flat], tf.int64)
    return ids


def num_to_char(ids: tf.Tensor) -> tf.Tensor:
    flat = tf.reshape(tf.cast(ids, tf.int64), (-1,))
    chars = tf.numpy_function(_np_ids_to_chars, [flat], tf.string)
    return chars


def load_video(path:str) -> List[float]: 
    #print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def preprocess_frames_to_75(frames: tf.Tensor) -> tf.Tensor:
    # Helper to pad or truncate to 75 frames for model inference
    num_frames = tf.shape(frames)[0]
    target_frames = 75
    def truncate():
        return frames[:target_frames]
    def pad():
        pad_len = target_frames - num_frames
        pad_tensor = tf.zeros((pad_len, tf.shape(frames)[1], tf.shape(frames)[2], tf.shape(frames)[3]), dtype=frames.dtype)
        return tf.concat([frames, pad_tensor], axis=0)
    return tf.cond(num_frames >= target_frames, truncate, pad)
    

def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    token_tensor = tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1))
    return char_to_num(token_tensor)[1:]


def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('..','data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('..','data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments