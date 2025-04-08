import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.utils import custom_object_scope
import wave
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from script import read_wav_as_float, write_float_samples_to_wav, build_rnn_spectrogram_model, SpectrogramModelLayer, STFTLayer, ISTFTLayer, AddInnerDim, Squeeze, snr_cost

file_for_restoration_path = "1c.wav"
SQNC_LENGTH = 512

def threshold_mask(reference,restored, threshold_ratio=0.95):
    """
    Create a binary mask for a numpy array based on the absolute maximum value.
    
    Parameters:
        x (np.ndarray): Input array of float values.
        threshold_ratio (float): Threshold ratio (default=0.95). Samples with an absolute value
                                 greater than or equal to threshold_ratio times the absolute maximum
                                 will be marked with 1, others with 0.
    
    Returns:
        np.ndarray: A binary mask of the same shape as x.
    """
    abs_x = np.abs(reference)
    max_val = np.max(abs_x)
    
    if max_val == 0:
        # If the maximum is 0, return an array of zeros to avoid division by zero.
        return np.zeros_like(reference, dtype=int)
    
    mask = (abs_x >= (threshold_ratio * max_val)).astype(int)
    return (1-mask)*reference + restored*mask


class CustomInputLayer(tf.keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        # Remove 'batch_shape' if present (or remap it to 'shape')
        if 'batch_shape' in config:
            # Option 1: Remove it entirely
            config.pop('batch_shape')
            # Option 2 (if needed): convert it to 'shape'
            # config['shape'] = config.pop('batch_shape')[1:]
        return super(CustomInputLayer, cls).from_config(config)


with wave.open(file_for_restoration_path, 'rb') as wav_file:
    fs = wav_file.getframerate()


model = tf.keras.models.load_model(
    "model_checkpoint_mae3.keras",
    custom_objects={
        'SpectrogramModelLayer': SpectrogramModelLayer,
        'STFTLayer': STFTLayer,
        'ISTFTLayer': ISTFTLayer,
        'AddInnerDim': AddInnerDim,
        'Squeeze': Squeeze,
        'InputLayer': CustomInputLayer,
        'DTypePolicy': Policy,
        'snr_cost': snr_cost
    }
)

samples_input_file = read_wav_as_float(file_for_restoration_path)
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
overlap_input_sequences = np.array(overlap_input_sequences)
nn_restored = model.predict_on_batch(overlap_input_sequences)
i = 0
for sqnc in overlap_input_sequences:
  if(max(sqnc)>(maxv*0.95) or min(sqnc)<(minv*0.95)):
    restored = nn_restored[i][SQNC_LENGTH//4:(SQNC_LENGTH*3)//4]
    reference = np.array(sqnc[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4])
    restored_samples_overlap.append(threshold_mask(reference, restored))
  else:
    restored_samples_overlap.append(np.array(sqnc[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4]))
  i += 1
restored_samples_overlap = np.array(restored_samples_overlap).flatten()
#print(type(restored_samples_overlap))
print(restored_samples_overlap.shape)
output_path = 'output12.wav'  # Path to save the WAV file

#write_float_samples_to_wav(samples_restored_final, fs, output_path)
#print(f"WAV file written to {output_path}")
restored_samples_overlap = np.array(restored_samples_overlap).flatten()
restored_samples_overlap = np.append(np.array(samples_input_file[0:SQNC_LENGTH//4]),restored_samples_overlap)
restored_samples_overlap = np.append(restored_samples_overlap,np.array(samples_input_file[-SQNC_LENGTH//4:]))
write_float_samples_to_wav(restored_samples_overlap, fs, output_path)
print(f"WAV file written to {output_path}")
