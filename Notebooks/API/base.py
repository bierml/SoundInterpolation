import traceback
import warnings
import sys
import contextlib
import wave_bwf_rf64
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import wave
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Multiply
SQNC_LENGTH = 512
def snr_cost(s_estimate, s_true):
    '''Static Method defining the cost function. 
    The negative signal to noise ratio is calculated here. The loss is 
    always calculated over the last dimension. 
    '''
   
    # calculating the SNR
    snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
    (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True)+1e-7)
    # using some more lines, because TF has no log10
    num = tf.math.log(snr) 
    denom = tf.math.log(tf.constant(10, dtype=num.dtype))
    loss = -10*(num / (denom))
    # returning the loss
    return loss

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

"""Функция для записи массива в файл по пути output_path."""

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

def get_number_of_samples(filename):
    with contextlib.closing(wave_bwf_rf64.open(filename, 'rb')) as wf:
        nframes = wf.getnframes()
    return nframes

def read_samples_segment(filename, start_index, end_index):
    """
    Reads and returns samples from an RF64 file between the specified frame indices.
    
    Parameters:
      filename (str): Path to the RF64 file.
      start_index (int): Starting frame index (inclusive).
      end_index (int): Ending frame index (exclusive).
    
    Returns:
      np.ndarray: An array of audio samples (reshaped to (num_frames, num_channels) if multichannel).
    """
    with contextlib.closing(wave_bwf_rf64.open(filename, 'rb')) as wf:
        nframes = wf.getnframes()
        #print("indicies,frames:",start_index,end_index,nframes)
        if start_index < 0 or end_index > nframes or start_index >= end_index:
            raise ValueError("Invalid start or end index.")
        
        # Set the file pointer to the desired start frame.
        wf.setpos(start_index)
        # Read the desired number of frames.
        frames = wf.readframes(end_index - start_index)
        
        sample_width = wf.getsampwidth()
        num_channels = wf.getnchannels()
        
        # Map sample width (in bytes) to numpy dtype.
        if sample_width == 1:
            dtype = np.uint8  # usually unsigned
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert the raw bytes to a numpy array.
        samples = np.frombuffer(frames, dtype=dtype)/(2**(sample_width*8-1))
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels)
        return samples
    
class AudioDataGenerator(Sequence):
    """
    A Keras Sequence that loads entire RF64 files into RAM and generates batches of overlapping
    audio sequences (with 50% overlap). Since all data are already in memory, __getitem__ only slices
    preloaded arrays.
    """
    def __init__(self, original_file, clipped_file, SQNC_LENGTH, batch_size = 32, cache_size = 100, shuffle=True):
        self.SQNC_LENGTH = SQNC_LENGTH
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ofname = original_file
        self.cfname = clipped_file
        self.cache_size = cache_size
        # Get total number of sample frames from one of the files.
        self.nofsamples = get_number_of_samples(original_file)
        # Compute overlapping sequence starting indices (50% overlap)
        self.step_size = SQNC_LENGTH // 2
        self.indices = list(range(0, self.nofsamples - SQNC_LENGTH + 1, self.step_size))
        #number of the latest sequence stored in cache
        self.startindex = -1
        self.endindex = -1
        self.cache_samples_orig = []
        self.cache_samples_clip = []
        if self.shuffle:
            for i in range(int(np.ceil(len(self.indices)/(self.batch_size*self.cache_size)))):
                cache = self.indices[self.batch_size*self.cache_size*i:self.batch_size*self.cache_size*(i+1)]
                np.random.shuffle(cache)
                self.indices[self.batch_size*self.cache_size*i:self.batch_size*self.cache_size*(i+1)] = cache

    def __len__(self):
        # Return the number of batches per epoch.
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        #print("idx=",idx)
        #check if updating the cache is needed
        if((idx//self.cache_size)*self.cache_size*self.batch_size != self.startindex):
           #updating the cache
           #number of batch for cache to start
           cachestart_in_batch_num = (idx//self.cache_size)*self.cache_size
           self.startindex = cachestart_in_batch_num*self.batch_size
           #cache size in batches
           cachesz = self.cache_size*self.batch_size
           #add (cache size-1) to startindex to find actual last sequence in cache
           self.endindex = self.startindex + cachesz - 1
           #print(self.startindex,self.endindex)
           #print("indicies before correction: ", self.startindex,self.endindex)
           #idx - number of currently requested batch of sequences 
           #startindex and endindex - indicies of the first (inclusively) and last (exclusively) sequences stored in cache
           #example: 3627th sequence -> startindex = 3600, endindex = 3699 (for cachesize=100)
           if(self.endindex > len(self.indices)-1):
               self.endindex = len(self.indices)-1
           #absolute index of cache starting sample in the whole file
           fsqstartabsolute = self.startindex*self.step_size
           #absolute index of the last cache sequence's start 
           lsqstartabsolute = self.endindex*self.step_size
           #absolute index of the last cache sequence's end
           lsqendabsolute = lsqstartabsolute + self.SQNC_LENGTH - 1
           self.cache_samples_orig = read_samples_segment(self.ofname,fsqstartabsolute,lsqendabsolute+1)
           self.cache_samples_clip = read_samples_segment(self.cfname,fsqstartabsolute,lsqendabsolute+1)
        # Get the global starting indices for this batch.
        start_batch = idx * self.batch_size
        end_batch = (idx + 1) * self.batch_size
        if(end_batch>len(self.indices)-1):
            end_batch= len(self.indices)-1
        batch_indices = self.indices[start_batch : end_batch]
        X_batch = []
        y_batch = []
        for start in batch_indices:
            # Slice a contiguous sequence of length SQNC_LENGTH from the preloaded arrays.
            #absolute index in the whole file where cache starts
            cachestartabs = self.startindex*self.step_size
            #print("idx:",idx)
            X_seq = self.cache_samples_clip[start - cachestartabs : start - cachestartabs + self.SQNC_LENGTH]
            y_seq = self.cache_samples_orig[start - cachestartabs : start - cachestartabs + self.SQNC_LENGTH]
            X_batch.append(X_seq)
            y_batch.append(y_seq)
        # Convert list of sequences to NumPy arrays.
        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        # Optionally shuffle starting indices at the end of each epoch.
        if self.shuffle:
            batch_indices = list(range(int(np.ceil(len(self.indices)/(self.batch_size*self.cache_size)))))
            np.random.shuffle(batch_indices)
            for i in batch_indices:
                cache = self.indices[self.batch_size*self.cache_size*i:self.batch_size*self.cache_size*(i+1)]
                np.random.shuffle(cache)
                self.indices[self.batch_size*self.cache_size*i:self.batch_size*self.cache_size*(i+1)] = cache

"""Обучение нейросети на множестве спектрограмм сигнала. N и M - количество точек по осям частоты и времени соответственно в обучающих выборках."""

import tensorflow as tf
FSTEP = 8
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
class SpectrogramModelLayer(tf.keras.layers.Layer):
    def __init__(self, sq_lngth, rnn_layer=tf.keras.layers.SimpleRNN, **kwargs):
        super(SpectrogramModelLayer, self).__init__(**kwargs)
        self.sq_lngth = sq_lngth
        self.rnn_layer = rnn_layer
        self.frame_step = FSTEP
        self.frame_length = FSTEP * 2  # e.g. if FSTEP=8 then frame_length=16
        # Frequency bins: frame_length//2 + 1 = FSTEP+1 (e.g. 9 if FSTEP=8)
        self.F_const = self.frame_step + 1  
        # Time frames computed from signal length:
        self.M_const = (sq_lngth - self.frame_length) // self.frame_step + 1

        # Instantiate custom STFT/ISTFT and helper layers.
        self.stft_layer = STFTLayer(frame_length=self.frame_length, frame_step=self.frame_step)
        self.istft_layer = ISTFTLayer(frame_length=self.frame_length, frame_step=self.frame_step, sq_lngth=sq_lngth)
        self.add_inner = AddInnerDim()
        self.squeeze = Squeeze()

        # Layers for processing the magnitude spectrogram.
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')
        self.rnn1 = self.rnn_layer(units=sq_lngth, return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(0.25)
        self.rnn2 = self.rnn_layer(units=sq_lngth, return_sequences=True)
        self.dense = tf.keras.layers.Dense(units=self.M_const, activation='linear')

        # Post-ISTFT refinement block:
        # Original pipeline: Conv1D -> SimpleRNN -> Dense -> (residual addition)
        # Now we add an extra SimpleRNN layer before the existing SimpleRNN and a second Conv1D after the Dense.
        self.convout = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
        # New additional SimpleRNN layer (rnnout2)
        self.rnnout2 = self.rnn_layer(units=32, return_sequences=True)
        # Existing SimpleRNN layer (rnnout)
        self.rnnout = self.rnn_layer(units=sq_lngth//2, return_sequences=True)
        # Change denseout to output 1 unit per timestep.
        self.denseout = tf.keras.layers.Dense(units=sq_lngth//2, activation='linear')
        # New additional Conv1D layer (convout2) before adding residual connection.
        self.convout2 = tf.keras.layers.Conv1D(filters=1, kernel_size=3, activation='linear', padding='same')
    def call(self, inputs):
        # inputs: (batch, sq_lngth)
        mag, phase = self.stft_layer(inputs)  # both: (batch, F, T)
        # Crop to M_const time frames.
        mag = mag[:, :, :self.M_const]
        phase = phase[:, :, :self.M_const]
        # Add a singleton channel dimension (for Conv2D)
        mag = self.add_inner(mag)      # (batch, F, T, 1)
        phase = self.add_inner(phase)  # (batch, F, T, 1)

        # Process magnitude with Conv2D layers.
        x = self.conv1(mag)
        x = self.conv2(x)
        batch_size = tf.shape(x)[0]
        # Reshape for RNN: treat frequency dimension (F) as timesteps; flatten T and channels.
        x = tf.reshape(x, [batch_size, self.F_const, self.M_const * 64])  # (batch, F, T*64)

        # Process with two SimpleRNN layers.
        x = self.rnn1(x)
        x = self.dropout(x)
        x = self.rnn2(x)
        # Map each timestep to M_const outputs.
        x = self.dense(x)  # now x: (batch, F, M_const)
        #x = self.add_inner(x)
        mag_ = self.squeeze(mag)
        x = mag_ + x
	
        # Process phase: remove channel dimension → (batch, F, M_const)
        phase = self.squeeze(phase)

        # Reconstruct time-domain signal via ISTFT.
        reconstructed = self.istft_layer([x, phase])  # (batch, sq_lngth)

        # === Post-ISTFT refinement block ===
        # Expand dims for Conv1D: (batch, sq_lngth, 1)
        reconstructed = tf.reshape(reconstructed, [batch_size, self.sq_lngth, 1])
        # First Conv1D.
        rec_proc1 = self.convout(reconstructed)  # (batch, sq_lngth, 32)
        # New additional SimpleRNN layer.
        rec_proc = self.rnnout2(rec_proc1)       # (batch, sq_lngth, 32)
        # Existing SimpleRNN layer.
        rec_proc = self.rnnout(rec_proc)          # (batch, sq_lngth, sq_lngth//2)
        # TimeDistributed Dense to map each timestep to 1 feature.
        rec_proc = self.denseout(rec_proc)        # (batch, sq_lngth, 1)
        # New additional Conv1D layer.
        rec_proc = self.convout2(rec_proc)        # (batch, sq_lngth, 1)
        rec_proc = tf.squeeze(rec_proc, axis=-1)  # (batch, sq_lngth)
        # === End refinement block ===
        # Residual connection.
        return inputs + rec_proc
    
    def get_config(self):
        config = super(SpectrogramModelLayer, self).get_config()
        # Store the rnn_layer as its class name
        config.update({
            "sq_lngth": self.sq_lngth,
            "rnn_layer": self.rnn_layer.__name__
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Map the string name back to the actual class.
        rnn_layer_name = config.pop("rnn_layer")
        if rnn_layer_name == "LSTM":
            config["rnn_layer"] = tf.keras.layers.LSTM
        elif rnn_layer_name == "SimpleRNN":
            config["rnn_layer"] = tf.keras.layers.SimpleRNN
        else:
            raise ValueError(f"Unsupported rnn_layer: {rnn_layer_name}")
        return cls(**config)

# Now reimplement build_rnn_spectrogram_model using Sequential.
def build_rnn_spectrogram_model(sq_lngth,layer):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(sq_lngth,)),
        SpectrogramModelLayer(sq_lngth=sq_lngth,rnn_layer=layer)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mse']
    )
    return model
    