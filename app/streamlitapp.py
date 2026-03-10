# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import time
import numpy as np
import tempfile
import cv2
import json

import tensorflow as tf 
from utils import load_data, num_to_char, load_video
from modelutil import load_model

# Set page config and light branding
st.set_page_config(page_title='Echometric - Lip Reading', page_icon='👄', layout='wide')
st.markdown('''<style>
.echometric-header {font-size: 28px; font-weight: 700; margin: 8px 0 16px 0;}
.echometric-subtle {color: #6b7280;}
.echometric-accent {color: #8b5cf6;}
</style>''', unsafe_allow_html=True)

PREFS_PATH = os.path.join(os.path.dirname(__file__), '.echometric_prefs.json')

def _load_prefs():
    try:
        if os.path.exists(PREFS_PATH):
            with open(PREFS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_prefs(prefs: dict):
    try:
        with open(PREFS_PATH, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

prefs = _load_prefs()

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Echometric')
    st.caption('Lip reading research interface')
    st.info('This application is originally developed from the LipNet deep learning model.')
    backend_default_index = 0 if prefs.get('backend') in (None, 'TensorFlow (Keras)') else 1
    backend = st.radio('Inference backend', ['TensorFlow (Keras)', 'TensorFlow Lite (CPU)'], index=backend_default_index)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_weights = os.path.join(project_root, 'models', 'model_weights.h5')
    current_env = os.environ.get('MODEL_WEIGHTS_PATH')
    initial_value = current_env if (current_env and os.path.isabs(current_env)) else default_weights
    weights_path = st.text_input('Keras weights path', value=initial_value)
    if weights_path:
        resolved = weights_path if os.path.isabs(weights_path) else os.path.normpath(os.path.join(project_root, weights_path))
        os.environ['MODEL_WEIGHTS_PATH'] = resolved
    tflite_path = st.text_input('TFLite model path', value=prefs.get('tflite_path', os.path.join(project_root, 'models', 'lipnet_dynamic.tflite')))

    # Hidden defaults replacing removed UI controls
    compare_both = False
    runs = 1
    apply_denoise = False
    apply_contrast = False
    batch_folder = ''
    run_batch = False
    run_batch_predict = False
    gen_limit = 0
    gen_btn = False

    if st.button('Save settings'):
        _save_prefs({
            'backend': backend,
            'weights_path': weights_path,
            'tflite_path': tflite_path
        })
        st.success('Settings saved')

st.title('LipNet Full Stack App') 

# File uploader for custom video
st.subheader('Upload Video')
uploaded_file = st.file_uploader('Upload a lip-reading video (.mpg or .mp4)', type=['mpg', 'mp4'], help='Upload a new video file for lip reading prediction')

if uploaded_file is not None:
    st.success(f'✅ File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)')
    # Immediate preview using the uploaded buffer
    try:
        uploaded_file.seek(0)
        st.video(uploaded_file)
    except Exception:
        pass
    
    # Debug info
    with st.expander("🔍 Debug Info"):
        st.write(f"**File type:** {uploaded_file.type}")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size:,} bytes")
        st.write(f"**File extension:** {os.path.splitext(uploaded_file.name)[1]}")
        
        # Test if we can read the file
        try:
            file_content = uploaded_file.read()
            st.write(f"**File content length:** {len(file_content):,} bytes")
            st.write("✅ File can be read successfully")
            # Reset file pointer
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"❌ Cannot read file: {e}")
    
if st.button('Clear Upload Cache'):
    st.cache_data.clear()
    st.rerun()

# Test video button
if st.button('🎬 Test Video Display'):
    st.write("Testing video display with a sample...")
    try:
        # Try to use the built-in test video
        test_video_path = os.path.join(os.path.dirname(__file__), 'test_video.mp4')
        if os.path.exists(test_video_path):
            st.video(test_video_path)
            st.success("✅ Test video is playing!")
        else:
            st.warning("No test video found. Please upload a video to test.")
    except Exception as e:
        st.error(f"Test video failed: {e}")

# Generating a list of options or videos from GRID corpus
options = []
data_dir = os.path.join('..', 'data', 's1')
if os.path.isdir(data_dir):
    options = os.listdir(data_dir)
selected_video = None
if options:
    selected_video = st.selectbox('Or choose a sample GRID video', options)

# Built-in demo samples (no alignments required)
demo_samples = []
demo_dir = os.path.join(os.path.dirname(__file__), 'samples')
try:
    if os.path.isdir(demo_dir):
        for f in os.listdir(demo_dir):
            if f.lower().endswith(('.mp4', '.mpg', '.avi')):
                demo_samples = [*demo_samples, os.path.join(demo_dir, f)]
    # Fallback to bundled test video in app directory if present
    bundled_test = os.path.join(os.path.dirname(__file__), 'test_video.mp4')
    if os.path.exists(bundled_test):
        demo_samples = [bundled_test, *demo_samples]
except Exception:
    demo_samples = []

selected_demo = None
if demo_samples:
    demo_labels = [os.path.basename(p) for p in demo_samples]
    choice = st.selectbox('Or use a built-in demo sample', ['(none)'] + demo_labels)
    if choice != '(none)':
        selected_demo = demo_samples[demo_labels.index(choice)]

# Generate two columns 
col1, col2 = st.columns(2)

def _ensure_75_frames(frames_tensor: tf.Tensor) -> tf.Tensor:
    # frames_tensor shape: (T, H, W, C)
    num_frames = tf.shape(frames_tensor)[0]
    target_frames = 75
    def truncate():
        return frames_tensor[:target_frames]
    def pad():
        pad_len = target_frames - num_frames
        pad_tensor = tf.zeros((pad_len, tf.shape(frames_tensor)[1], tf.shape(frames_tensor)[2], tf.shape(frames_tensor)[3]), dtype=frames_tensor.dtype)
        return tf.concat([frames_tensor, pad_tensor], axis=0)
    return tf.cond(num_frames >= target_frames, truncate, pad)

def _apply_preprocessing(frames_tensor: tf.Tensor) -> tf.Tensor:
    # frames_tensor: (T, H, W, C) float32
    frames = frames_tensor
    if apply_denoise:
        # Apply Gaussian blur per frame
        frames_list = []
        for i in range(frames.shape[0]):
            f = frames[i]
            f_np = f.numpy()
            f_np = cv2.GaussianBlur(f_np, (3,3), 0)
            frames_list.append(tf.convert_to_tensor(f_np))
        frames = tf.stack(frames_list, axis=0)
    if apply_contrast:
        # Apply CLAHE per frame on grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        frames_list = []
        for i in range(frames.shape[0]):
            f = frames[i]
            f_np = f.numpy()
            f_np = np.uint8(np.clip((f_np - f_np.min()) / (f_np.max() - f_np.min() + 1e-6) * 255.0, 0, 255))
            f_np = clahe.apply(f_np.squeeze()).reshape(f_np.shape[0], f_np.shape[1], 1)
            f_np = f_np.astype(np.float32) / 255.0
            frames_list.append(tf.convert_to_tensor(f_np))
        frames = tf.stack(frames_list, axis=0)
    return frames

def _apply_preprocessing_flags(frames_tensor: tf.Tensor, use_denoise: bool, use_contrast: bool) -> tf.Tensor:
    frames = frames_tensor
    if use_denoise:
        frames_list = []
        for i in range(frames.shape[0]):
            f_np = frames[i].numpy()
            f_np = cv2.GaussianBlur(f_np, (3,3), 0)
            frames_list.append(tf.convert_to_tensor(f_np))
        frames = tf.stack(frames_list, axis=0)
    if use_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        frames_list = []
        for i in range(frames.shape[0]):
            f_np = frames[i].numpy()
            f_np = np.uint8(np.clip((f_np - f_np.min()) / (f_np.max() - f_np.min() + 1e-6) * 255.0, 0, 255))
            f_np = clahe.apply(f_np.squeeze()).reshape(f_np.shape[0], f_np.shape[1], 1)
            f_np = f_np.astype(np.float32) / 255.0
            frames_list.append(tf.convert_to_tensor(f_np))
        frames = tf.stack(frames_list, axis=0)
    return frames

@st.cache_data(show_spinner=False)
def _preprocess_cached(path: str, mtime: float, use_denoise: bool, use_contrast: bool) -> tf.Tensor:
    vt = load_video(path)
    vt = _apply_preprocessing_flags(vt, use_denoise, use_contrast)
    vt = _ensure_75_frames(vt)
    return vt


# Decide input source and render/predict
file_path = None
source_label = None
if uploaded_file is not None:
    # Save uploaded file to a temp location with unique name
    try:
        file_id = str(uploaded_file.name) + str(time.time())
        temp_suffix = os.path.splitext(uploaded_file.name)[1]
        if not temp_suffix:
            temp_suffix = '.mp4'  # Default to mp4 if no extension
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix, prefix=f"upload_{hash(file_id)}_") as tmp:
            uploaded_file.seek(0)  # Reset file pointer
            tmp.write(uploaded_file.read())
            file_path = tmp.name
            source_label = 'Uploaded video'
            st.write(f"📁 Saved to: {file_path}")
        
        # If not mp4, attempt to transcode to mp4 for browser playback
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ('.mp4',):
            try:
                mp4_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', prefix='upload_transcoded_')
                mp4_tmp.close()
                cmd = f"ffmpeg -y -loglevel error -i \"{file_path}\" -vcodec libx264 -pix_fmt yuv420p -movflags +faststart \"{mp4_tmp.name}\""
                rc = os.system(cmd)
                if rc == 0 and os.path.exists(mp4_tmp.name):
                    file_path = mp4_tmp.name
                    st.info('Converted to mp4 for browser compatibility.')
            except Exception as _transcode_err:
                pass
    except Exception as e:
        st.error(f'❌ Failed to save uploaded file: {e}')
        file_path = None
elif selected_demo:
    file_path = selected_demo
    source_label = 'Built-in demo sample'
elif selected_video:
    file_path = os.path.join('..','data','s1', selected_video)
    source_label = 'Sample GRID video'

if file_path:

    # Rendering the video 
    with col1: 
        st.info(f'The video below is used for prediction ({source_label})')
        # Working video preview using HTML5
        st.write("**Video Preview:**")
        
        # Check if file exists and show info
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            st.write(f"📁 File: {os.path.basename(file_path)}")
            st.write(f"📊 Size: {file_size:,} bytes")
            
            # Method 1: Try Streamlit's built-in video player
            try:
                st.video(file_path)
                st.success("✅ Video is now playing!")
            except Exception as e:
                st.write(f"Streamlit video failed: {e}")
                
                # Method 2: HTML5 video player (most reliable)
                try:
                    import base64
                    with open(file_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        video_base64 = base64.b64encode(video_bytes).decode()
                        
                        # Determine MIME type based on file extension
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext == '.mp4':
                            mime_type = 'video/mp4'
                        elif file_ext == '.mpg' or file_ext == '.mpeg':
                            mime_type = 'video/mpeg'
                        else:
                            mime_type = 'video/mp4'  # Default
                        
                        video_html = f"""
                        <div style="text-align: center;">
                            <video width="100%" height="400" controls autoplay muted>
                                <source src="data:{mime_type};base64,{video_base64}" type="{mime_type}">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                        """
                        st.markdown(video_html, unsafe_allow_html=True)
                        st.success("✅ Video is now playing with HTML5 player!")
                        
                except Exception as e2:
                    st.error(f"❌ HTML5 video failed: {e2}")
                    
                    # Method 3: Show first frame as fallback
                    try:
                        import cv2
                        cap = cv2.VideoCapture(file_path)
                        ret, frame = cap.read()
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, caption="First frame of video", width=400)
                            st.info("📸 Showing first frame - video processing will still work")
                        else:
                            st.warning("⚠️ Could not read video frames")
                        cap.release()
                    except Exception as e3:
                        st.error(f"❌ Could not extract frame: {e3}")
                        st.info("💡 Video file is loaded and ready for lip reading prediction")
        else:
            st.error(f"❌ File not found: {file_path}")


    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')
        if uploaded_file is not None or selected_demo is not None:
            # Use cached preprocess for uploads or built-in demos (includes 75-frame norm)
            video_tensor = _preprocess_cached(file_path, os.path.getmtime(file_path), apply_denoise, apply_contrast)
        else:
            # Preserve original pipeline for GRID samples but add caching and toggles
            video_tensor_raw, _ = load_data(tf.convert_to_tensor(file_path))
            video_tensor_raw_path_key = file_path  # keying by path and mtime
            video_tensor = _apply_preprocessing_flags(video_tensor_raw, apply_denoise, apply_contrast)
            video_tensor = _ensure_75_frames(video_tensor)

        # Prepare grayscale frames for GIF: squeeze channel and scale to uint8
        try:
            frames_np = np.squeeze(video_tensor.numpy(), axis=-1)
            frames_min = frames_np.min()
            frames_ptp = frames_np.max() - frames_min + 1e-6
            frames_uint8 = ((frames_np - frames_min) / frames_ptp * 255.0).astype(np.uint8)
            imageio.mimsave('animation.gif', frames_uint8, fps=10)
            st.image('animation.gif', width=400)
        except Exception as e:
            st.warning(f'Could not render GIF preview: {e}')

        # Frame scrubber to inspect individual frames
        with st.expander('Inspect frames'):
            idx = st.slider('Frame index', min_value=0, max_value=74, value=0, step=1)
            # Safeguard for short sequences
            idx = min(idx, int(video_tensor.shape[0]) - 1)
            frame = np.squeeze(video_tensor[idx].numpy(), axis=-1)
            st.image(frame, caption=f'Frame {idx}', clamp=True)

        @st.cache_resource(show_spinner=False)
        def get_model():
            try:
                st.info(f"Loading model from: {os.environ.get('MODEL_WEIGHTS_PATH', 'models/ckpt96/checkpoint')}")
                model = load_model()
                st.success("✅ Model loaded successfully!")
                return model
            except Exception as e:
                st.error(f'Model failed to load: {e}')
                st.error('Ensure weights exist at the correct path.')
                raise

        @st.cache_resource(show_spinner=False)
        def get_tflite_interpreter(model_path: str):
            if not os.path.exists(model_path):
                st.error(f'TFLite model not found at {model_path}. Generate it via optimize_model.py.')
                raise FileNotFoundError(model_path)
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            return interpreter

        def run_keras(vt: tf.Tensor, r: int) -> tuple:
            try:
                st.write("Loading Keras model...")
                model = get_model()
                times = []
                yhat_local = None
                st.write("Running Keras prediction...")
                for i in range(r):
                    t0 = time.time()
                    yhat_local = model.predict(tf.expand_dims(vt, axis=0))
                    times.append((time.time() - t0) * 1000.0)
                    st.write(f"Run {i+1}/{r} completed")
                st.success("Keras prediction completed!")
                return yhat_local, float(np.mean(times)), float(np.median(times))
            except Exception as e:
                st.error(f"Keras prediction failed: {e}")
                st.exception(e)
                raise

        def run_tflite(vt: tf.Tensor, r: int, path: str) -> tuple:
            try:
                st.write("Initializing TFLite interpreter...")
                interpreter = get_tflite_interpreter(path)
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                st.write("Preparing input data...")
                input_data = tf.expand_dims(vt, axis=0)
                input_data = tf.cast(input_data, input_details[0]['dtype'])
                
                times = []
                yhat_local = None
                st.write("Running prediction...")
                for i in range(r):
                    interpreter.set_tensor(input_details[0]['index'], input_data.numpy())
                    t0 = time.time()
                    interpreter.invoke()
                    times.append((time.time() - t0) * 1000.0)
                    yhat_local = interpreter.get_tensor(output_details[0]['index'])
                    st.write(f"Run {i+1}/{r} completed")
                
                st.success("TFLite prediction completed!")
                return yhat_local, float(np.mean(times)), float(np.median(times))
            except Exception as e:
                st.error(f"TFLite prediction failed: {e}")
                st.exception(e)
                raise

        def decode_and_summarize(yhat_arr) -> tuple:
            decoder = tf.keras.backend.ctc_decode(yhat_arr, [75], greedy=True)[0][0].numpy()
            tokens_str = str(decoder)
            text = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            # Simple confidence proxy: mean of max probabilities per timestep
            probs = tf.nn.softmax(yhat_arr, axis=-1).numpy()
            step_conf = probs.max(axis=-1).mean()
            return tokens_str, text, float(step_conf)

        if compare_both:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader('TensorFlow (Keras)')
                try:
                    yhat_k, mean_ms_k, med_ms_k = run_keras(video_tensor, runs)
                    tokens_k, text_k, conf_k = decode_and_summarize(yhat_k)
                    st.text(tokens_k)
                    st.text(text_k)
                    st.caption(f'Avg: {mean_ms_k:.1f} ms, Median: {med_ms_k:.1f} ms, Confidence≈ {conf_k:.2f}')
                    st.download_button('Download prediction (Keras)', data=text_k, file_name='prediction_keras.txt')
                except Exception as e:
                    st.exception(e)
            with col_b:
                st.subheader('TensorFlow Lite (CPU)')
                try:
                    yhat_t, mean_ms_t, med_ms_t = run_tflite(video_tensor, runs, tflite_path)
                    tokens_t, text_t, conf_t = decode_and_summarize(yhat_t)
                    st.text(tokens_t)
                    st.text(text_t)
                    st.caption(f'Avg: {mean_ms_t:.1f} ms, Median: {med_ms_t:.1f} ms, Confidence≈ {conf_t:.2f}')
                    st.download_button('Download prediction (TFLite)', data=text_t, file_name='prediction_tflite.txt')
                except Exception as e:
                    st.exception(e)
        else:
            st.info('This is the output of the machine learning model as tokens')
            try:
                st.write(f"**Backend:** {backend}")
                st.write(f"**Video tensor shape:** {video_tensor.shape}")
                
                if backend == 'TensorFlow Lite (CPU)':
                    st.write("Running TensorFlow Lite prediction...")
                    yhat, mean_ms, med_ms = run_tflite(video_tensor, runs, tflite_path)
                else:
                    st.write("Running TensorFlow Keras prediction...")
                    yhat, mean_ms, med_ms = run_keras(video_tensor, runs)
                
                st.write(f"**Prediction shape:** {yhat.shape}")
                tokens, text, conf = decode_and_summarize(yhat)
                
                st.subheader("Raw Tokens:")
                st.text(tokens)
                st.subheader("Decoded Text:")
                st.text(text)
                st.caption(f'Avg: {mean_ms:.1f} ms, Median: {med_ms:.1f} ms using {backend} | Confidence≈ {conf:.2f}')
                st.download_button('Download prediction', data=text, file_name='prediction.txt')
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)

    # Batch benchmarking section
    if run_batch and batch_folder:
        st.subheader('Batch Benchmark Results')
        video_files = []
        try:
            for f in os.listdir(batch_folder):
                if f.lower().endswith(('.mp4', '.avi', '.mpg')):
                    video_files.append(os.path.join(batch_folder, f))
        except Exception as e:
            st.error(f'Failed to list folder: {e}')
            video_files = []
        if not video_files:
            st.warning('No videos found in the specified folder.')
        else:
            rows = []
            try:
                for vf in video_files:
                    vt = load_video(vf)
                    vt = _apply_preprocessing(vt)
                    vt = _ensure_75_frames(vt)
                    result = {'file': os.path.basename(vf)}
                    if compare_both:
                        try:
                            _, mean_ms_k, med_ms_k = run_keras(vt, runs)
                            result.update({'keras_avg_ms': round(mean_ms_k, 2), 'keras_med_ms': round(med_ms_k, 2)})
                        except Exception as e:
                            result.update({'keras_avg_ms': None, 'keras_med_ms': None})
                        try:
                            _, mean_ms_t, med_ms_t = run_tflite(vt, runs, tflite_path)
                            result.update({'tflite_avg_ms': round(mean_ms_t, 2), 'tflite_med_ms': round(med_ms_t, 2)})
                        except Exception as e:
                            result.update({'tflite_avg_ms': None, 'tflite_med_ms': None})
                    else:
                        try:
                            if backend == 'TensorFlow Lite (CPU)':
                                _, mean_ms, med_ms = run_tflite(vt, runs, tflite_path)
                            else:
                                _, mean_ms, med_ms = run_keras(vt, runs)
                            result.update({'avg_ms': round(mean_ms, 2), 'med_ms': round(med_ms, 2), 'backend': backend})
                        except Exception as e:
                            result.update({'avg_ms': None, 'med_ms': None, 'backend': backend})
                    rows.append(result)
            except Exception as e:
                st.exception(e)
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button('Download results CSV', data=csv, file_name='benchmark_results.csv', mime='text/csv')

    # Batch prediction section (outputs text predictions CSV)
    if run_batch_predict and batch_folder:
        st.subheader('Batch Predictions')
        video_files = []
        try:
            for f in os.listdir(batch_folder):
                if f.lower().endswith(('.mp4', '.avi', '.mpg')):
                    video_files.append(os.path.join(batch_folder, f))
        except Exception as e:
            st.error(f'Failed to list folder: {e}')
            video_files = []
        if not video_files:
            st.warning('No videos found in the specified folder.')
        else:
            rows = []
            try:
                for vf in video_files:
                    vt = load_video(vf)
                    vt = _apply_preprocessing_flags(vt, apply_denoise, apply_contrast)
                    vt = _ensure_75_frames(vt)
                    if backend == 'TensorFlow Lite (CPU)':
                        yhat, _, _ = run_tflite(vt, 1, tflite_path)
                        bname = 'tflite'
                    else:
                        yhat, _, _ = run_keras(vt, 1)
                        bname = 'keras'
                    tokens, text, conf = decode_and_summarize(yhat)
                    rows.append({'file': os.path.basename(vf), 'backend': bname, 'prediction': text, 'confidence': round(conf, 4)})
            except Exception as e:
                st.exception(e)
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv', mime='text/csv')

    # Generate lips-only demo samples from GRID
    if 'gen_btn' in locals() and gen_btn:
        grid_dir = os.path.join('..', 'data', 's1')
        out_dir = os.path.join(os.path.dirname(__file__), 'samples')
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.isdir(grid_dir):
            st.error('GRID folder not found at ../data/s1. Please add GRID videos to generate demos.')
        else:
            created = 0
            for fname in sorted(os.listdir(grid_dir)):
                if not fname.lower().endswith('.mpg'):
                    continue
                in_path = os.path.join(grid_dir, fname)
                try:
                    cap = cv2.VideoCapture(in_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    width = 140
                    height = 46
                    scale = 3
                    out_w, out_h = width * scale, height * scale
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_path = os.path.join(out_dir, f'lips_only_{os.path.splitext(fname)[0]}.mp4')
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h), isColor=False)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        crop = gray[190:236, 80:220]  # same crop as model input
                        up = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                        writer.write(cv2.cvtColor(up, cv2.COLOR_GRAY2BGR))
                    cap.release()
                    writer.release()
                    created += 1
                except Exception as e:
                    st.warning(f'Failed to convert {fname}: {e}')
                if created >= int(gen_limit):
                    break
            st.success(f'Created {created} lips-only demo video(s) in app/samples/. Select them from the built-in demo dropdown.')