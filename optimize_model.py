import os
import tensorflow as tf

# Reuse existing model builder and weights loading
from app.modelutil import load_model


def convert_to_tflite_dynamic(model: tf.keras.Model, output_path: str) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enable Select TF ops to support TensorList/LSTM graphs; disable lowering
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False  # noqa: SLF001
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


def main():
    model = load_model()

    os.makedirs('models', exist_ok=True)
    output_tflite = os.path.join('models', 'lipnet_dynamic.tflite')

    convert_to_tflite_dynamic(model, output_tflite)

    # Size reporting
    keras_weights_path = os.environ.get('MODEL_WEIGHTS_PATH', os.path.join('..','models','checkpoint'))
    tflite_size_mb = os.path.getsize(output_tflite) / (1024 * 1024)
    print(f'Saved TFLite model to: {output_tflite} ({tflite_size_mb:.2f} MB)')
    print(f'Weights were loaded from: {keras_weights_path}')


if __name__ == '__main__':
    main()


