from base import *
import sys
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.utils import custom_object_scope
if __name__ == '__main__':
    if(len(sys.argv)<5):
        print('No enough arguments passed!')
    else:
        file_for_restoration_path = sys.argv[2]
        model_for_load_path = sys.argv[1]
        output_path = sys.argv[3]  # Path to save the WAV file
        alpha = float(sys.argv[4])
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
            model_for_load_path,
            custom_objects={
                'SpectrogramModelLayer': SpectrogramModelLayer,
                'InputLayer': CustomInputLayer,
                'DTypePolicy': Policy,
                'mse_shortened': mse_shortened
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
            if(j+SQNC_LENGTH < len(samples_input_file)):
                overlap_input_sequences.append(samples_input_file[j:j+SQNC_LENGTH])
            j += step_size
        overlap_input_sequences = np.array(overlap_input_sequences)
        nn_restored = model.predict_on_batch(overlap_input_sequences)
        i = 0
        for sqnc in overlap_input_sequences:
            if(max(sqnc)>(maxv*alpha) or min(sqnc)<(minv*alpha)):
                restored = nn_restored[i][SQNC_LENGTH//4:(SQNC_LENGTH*3)//4]
                reference = np.array(sqnc[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4])
                restored_samples_overlap.append(threshold_mask(reference, restored, alpha))
            else:
                restored_samples_overlap.append(np.array(sqnc[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4]))
            i += 1
        restored_samples_overlap = np.array(restored_samples_overlap).flatten()
        
        
        add = len(samples_input_file) - SQNC_LENGTH//4 - restored_samples_overlap.shape[0]
        restored_samples_overlap = np.append(np.array(samples_input_file[0:SQNC_LENGTH//4]),restored_samples_overlap)
        #restored_samples_overlap = np.append(restored_samples_overlap,np.array(samples_input_file[-SQNC_LENGTH//4:]))
        restored_samples_overlap = np.append(restored_samples_overlap,np.array(samples_input_file[-add:]))
        write_float_samples_to_wav(restored_samples_overlap, fs, output_path)
        print(f"WAV file written to {output_path}")



