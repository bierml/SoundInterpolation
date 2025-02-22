import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import wave
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

def read_wav_as_float(file_path):
    """
    Reads a WAV file and returns its samples as a NumPy array of float32 values.

    Parameters:
        file_path (str): Path to the WAV file.

    Returns:
        np.ndarray: An array of float32 samples in the range [-1.0, 1.0].
    """
    with wave.open(file_path, 'rb') as wav_file:
        # Get parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        print(f"Channels: {n_channels}, Sample Width: {sample_width}, Frame Rate: {frame_rate}, Frames: {n_frames}")

        # Read frames as bytes
        raw_data = wav_file.readframes(n_frames)

    # Determine the data type based on sample width
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Convert raw bytes to numpy array without copying data
    int_data = np.frombuffer(raw_data, dtype=dtype)

    # Convert to float32 and normalize to range [-1.0, 1.0]
    max_val = float(2 ** (8 * sample_width - 1))
    float_data = int_data.astype(np.float32) / max_val

    # Handle multi-channel audio by averaging channels
    if n_channels > 1:
        float_data = float_data.reshape(-1, n_channels).mean(axis=1)

    return float_data

SQNC_LENGTH = 256
FSTEP = 16
# Custom STFT layer using tf.signal.stft
class STFTLayer(tf.keras.layers.Layer):
    def __init__(self, frame_length=8, frame_step=4, **kwargs):
        super(STFTLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step

    def call(self, inputs):
        # inputs: shape (batch, sq_lngth)
        # Use a Hann window
        window = tf.signal.hann_window(self.frame_length, dtype=inputs.dtype)
        stft_result = tf.signal.stft(
            inputs,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            window_fn=lambda fl, dtype: window
        )
        # tf.signal.stft returns shape (batch, time_frames, fft_unique_bins)
        # For our design, we want to use (batch, fft_unique_bins, time_frames)
        magnitude = tf.abs(stft_result)
        phase = tf.math.angle(stft_result)
        # Transpose to shape (batch, fft_unique_bins, time_frames)
        magnitude = tf.transpose(magnitude, perm=[0, 2, 1])
        phase = tf.transpose(phase, perm=[0, 2, 1])
        return magnitude, phase

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        if input_shape[1] is None:
            return (batch, None, None), (batch, None, None)
        # time_frames computed from signal length:
        time_frames = (input_shape[1] - self.frame_length) // self.frame_step + 1
        fft_bins = self.frame_length // 2 + 1
        # After transposition, output shape becomes (batch, fft_bins, time_frames)
        return (batch, fft_bins, time_frames), (batch, fft_bins, time_frames)


# Custom inverse STFT layer using tf.signal.inverse_stft
class ISTFTLayer(tf.keras.layers.Layer):
    def __init__(self, frame_length=8, frame_step=4, sq_lngth=None, **kwargs):
        super(ISTFTLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.sq_lngth = sq_lngth

    def call(self, inputs):
        # inputs: a list [mag, phase] with shapes (batch, F, T)
        mag, phase = inputs
	# tf.signal.inverse_stft expects input of shape (batch, time_frames, fft_unique_bins).
	# So transpose mag and phase from (batch, F, T) to (batch, T, F):
        mag_t = tf.transpose(mag, perm=[0, 2, 1])
        phase_t = tf.transpose(phase, perm=[0, 2, 1])
        phase_float = tf.cast(phase_t, tf.float32)
        stft_complex = tf.cast(mag_t, tf.complex64) * tf.complex(tf.cos(phase_float), tf.sin(phase_float))
        #stft_complex = tf.cast(mag_t, tf.complex64) * tf.exp(1j * tf.cast(phase_t, tf.complex64))
        window = tf.signal.hann_window(self.frame_length, dtype=tf.float32)
        reconstructed = tf.signal.inverse_stft(
            stft_complex,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            window_fn=lambda fl, dtype: window
        )
        if self.sq_lngth is not None:
            reconstructed = reconstructed[:, :self.sq_lngth]
        return reconstructed

    def compute_output_shape(self, input_shape):
        batch = input_shape[0][0]
        if self.sq_lngth is not None:
            return (batch, self.sq_lngth)
        else:
            return (batch, None)


# Helper layers to add and remove a singleton channel dimension.
class AddInnerDim(tf.keras.layers.Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=-1)

class Squeeze(tf.keras.layers.Layer):
    def call(self, x):
        return tf.squeeze(x, axis=-1)

# Custom layer wrapping the entire spectrogram processing pipeline.
@tf.keras.utils.register_keras_serializable()
class SpectrogramModelLayer(tf.keras.layers.Layer):
    def __init__(self, sq_lngth, **kwargs):
        super(SpectrogramModelLayer, self).__init__(**kwargs)
        self.sq_lngth = sq_lngth
        self.frame_step = FSTEP
        self.frame_length = FSTEP * 2  # same as in original (8 if FSTEP=4)
        # Frequency bins: frame_length//2 + 1 = FSTEP+1 (e.g. 5)
        self.F_const = self.frame_step + 1
        # Time frames computed from signal length (same as original)
        self.M_const = (sq_lngth - self.frame_length) // self.frame_step + 1

        # Instantiate our custom STFT/ISTFT and helper layers.
        self.stft_layer = STFTLayer(frame_length=self.frame_length, frame_step=self.frame_step)
        self.istft_layer = ISTFTLayer(frame_length=self.frame_length, frame_step=self.frame_step, sq_lngth=sq_lngth)
        self.add_inner = AddInnerDim()
        self.squeeze = Squeeze()

        # Convolution and RNN layers for processing the magnitude.
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        # After Conv2D, the tensor shape is (batch, F, T, 64)
        # We reshape it to (batch, F, T*64) where F = F_const.
        self.rnn1 = tf.keras.layers.SimpleRNN(units=sq_lngth, activation='relu', return_sequences=True)
        self.rnn2 = tf.keras.layers.SimpleRNN(units=sq_lngth // 2, activation='relu', return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=self.M_const, activation='linear')

    def call(self, inputs):
        # inputs: shape (batch, sq_lngth)
        # Compute STFT; stft_layer returns a tuple (mag, phase) each with shape (batch, F, T)
        mag, phase = self.stft_layer(inputs)

        # Crop to M_const time frames (if necessary)
        mag = mag[:, :, :self.M_const]
        phase = phase[:, :, :self.M_const]

        # Add a singleton channel dimension (for Conv2D)
        mag = self.add_inner(mag)   # now shape: (batch, F, T, 1)
        phase = self.add_inner(phase)  # (batch, F, T, 1)

        # Process the magnitude with two Conv2D layers.
        x = self.conv1(mag)
        x = self.conv2(x)  # shape: (batch, F, T, 64)

        # Reshape for RNN processing:
        batch_size = tf.shape(x)[0]
        # We treat the frequency dimension F as the timesteps (F_const = FSTEP+1)
        # and flatten the T (time frames) and channel dimensions.
        x = tf.reshape(x, [batch_size, self.F_const, self.M_const * 64])

        # Process with two SimpleRNN layers.
        x = self.rnn1(x)
        x = self.rnn2(x)
        # Map each of the F timesteps to M_const outputs.
        x = self.dense(x)  # now x has shape (batch, F, M_const)

        # Process phase: remove the singleton channel dimension.
        phase = self.squeeze(phase)  # shape: (batch, F, M_const)

        # Reconstruct the time-domain signal via ISTFT.
        reconstructed = self.istft_layer([x, phase])  # shape: (batch, sq_lngth)
        # Add a residual connection: original input + reconstruction.
        return inputs + reconstructed




"""Загружаем модель с готовыми весами из файла"""
model_path = "clipping_interpolation_model.keras"
model = keras.saving.load_model(model_path, custom_objects={'SpectrogramModelLayer': SpectrogramModelLayer})

"""Для правильного восстановления нужны накладывающиеся последовательности семплов исходного файла. Для простоты возьмем степень наложения окон равной 0.5."""

#print(samples_restored)
file_for_restoration_path = "1cn.wav"
samples_input_file = read_wav_as_float(file_for_restoration_path)

with wave.open(file_for_restoration_path, 'rb') as wav_file:
    fs = wav_file.getframerate()
    
restored_samples_overlap = []
overlap_input_sequences = []
step_size = SQNC_LENGTH // 2
j = 0
maxv = np.max(np.array(samples_input_file))
minv = np.min(np.array(samples_input_file))
while j < len(samples_input_file):
    #print(j, j+SQNC_LENGTH-1)
    if(j+step_size < len(samples_input_file)):
        overlap_input_sequences.append(samples_input_file[j:j+SQNC_LENGTH])
    j += step_size
for sqnc in overlap_input_sequences:
  if(max(sqnc)>(maxv*0.95) or min(sqnc)<(minv*0.95)):
    elem = np.array(sqnc)
    elem = np.expand_dims(elem, axis=0)  # Now shape is (1, SQNC_LENGTH)
    res = model.predict(elem,verbose=0).flatten()
    #print(res[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4])
    restored_samples_overlap.append(res[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4])
  else:
    restored_samples_overlap.append(np.array(sqnc[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4]))

restored_samples_overlap = np.array(restored_samples_overlap).flatten()
print(restored_samples_overlap.shape)

"""Если мы хотим произвести сравнение с каким-либо другим методом, возможно, возникнет проблема из-за разных длин файлов: текущий алгоритм отбрасывает последние сэмплы в файле чтобы достичь количества сэмплов кратного SQNC_LENGTH. Если раскомментировать вторую строку мы получим массив в котором недостающие восстановленные сэмплы заменены сэмплами исходного массива до требуемой длины, что обеспечит возможность сравнения файлов. output_path - название файла, в который будет записан вывод программы."""


def write_float_samples_to_wav(samples, sample_rate, output_path):
    """
    Writes floating-point audio samples to a mono 16-bit WAV file.

    Parameters:
        samples (list or np.ndarray): Array of floating-point audio samples in the range [-1.0, 1.0].
        sample_rate (int): Sample rate of the audio in Hz (e.g., 44100).
        output_path (str): Path to save the output WAV file.
    """
    # Ensure the samples are a NumPy array
    samples = np.array(samples, dtype=np.float32)

    # Clip the samples to the range [-1.0, 1.0] to prevent overflow
    samples = np.clip(samples, -1.0, 1.0)

    # Convert to 16-bit PCM format
    int_samples = (samples * 32767).astype(np.int16)

    # Write to a WAV file
    with wave.open(output_path, 'wb') as wav_file:
        # Set the parameters for the WAV file
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)

        # Write the audio frames
        wav_file.writeframes(int_samples.tobytes())

output_path = 'output.wav'  # Path to save the WAV file

#write_float_samples_to_wav(samples_restored_final, fs, output_path)
#print(f"WAV file written to {output_path}")
restored_samples_overlap = np.array(restored_samples_overlap).flatten()
restored_samples_overlap = np.append(np.array(samples_input_file[0:SQNC_LENGTH//4]),restored_samples_overlap)
restored_samples_overlap = np.append(restored_samples_overlap,np.array(samples_input_file[-SQNC_LENGTH//4:]))
write_float_samples_to_wav(restored_samples_overlap, fs, output_path)
print(f"WAV file written to {output_path}")