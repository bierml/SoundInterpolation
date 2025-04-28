from base import *
if __name__ == '__main__':
    if(len(sys.argv)<4):
        print('No enough arguments passed!')
    else:
        wav_file_path = sys.argv[1]
        wav_file_path1 = sys.argv[2]
        # Example usage:
        # Assume SQNC_LENGTH, samples_sequences_clipped, and samples_sequences are defined.
        model = build_rnn_spectrogram_model(SQNC_LENGTH,tf.keras.layers.SimpleRNN,'relu')
        model.summary()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        batch_size = int(sys.argv[3])
        train_gen = AudioDataGenerator(wav_file_path, wav_file_path1, SQNC_LENGTH, batch_size=batch_size, cache_size=20, shuffle=True)
        checkpointer = ModelCheckpoint('model_checkpoint.keras',monitor='mse',verbose=1,save_best_only=True,save_weights_only=False,mode='min',save_freq='epoch')
        model.fit(train_gen,
                  verbose=1,
                  epochs=1000,
                  callbacks=[early_stopping,checkpointer])
