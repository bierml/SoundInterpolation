import tensorflow as tf

import wave
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from script import read_wav_as_float, write_float_samples_to_wav, build_rnn_spectrogram_model, SpectrogramModelLayer, STFTLayer, ISTFTLayer, AddInnerDim, Squeeze

file_for_restoration_path = "1c.wav"
SQNC_LENGTH = 256

with wave.open(file_for_restoration_path, 'rb') as wav_file:
    fs = wav_file.getframerate()


model = tf.keras.models.load_model(
    "clipping_interpolation_model.keras",
    custom_objects={
        'SpectrogramModelLayer': SpectrogramModelLayer,
        'STFTLayer': STFTLayer,
        'ISTFTLayer': ISTFTLayer,
        'AddInnerDim': AddInnerDim,
        'Squeeze': Squeeze
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
#print(type(restored_samples_overlap))
print(restored_samples_overlap.shape)
output_path = 'output.wav'  # Path to save the WAV file

#write_float_samples_to_wav(samples_restored_final, fs, output_path)
#print(f"WAV file written to {output_path}")
restored_samples_overlap = np.array(restored_samples_overlap).flatten()
restored_samples_overlap = np.append(np.array(samples_input_file[0:SQNC_LENGTH//4]),restored_samples_overlap)
restored_samples_overlap = np.append(restored_samples_overlap,np.array(samples_input_file[-SQNC_LENGTH//4:]))
write_float_samples_to_wav(restored_samples_overlap, fs, output_path)
print(f"WAV file written to {output_path}")
