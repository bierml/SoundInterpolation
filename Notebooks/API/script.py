from base import *  #базовый набор методов и объявлений для обучения и применения нейросети
if __name__ == '__main__':
    #недостаточно аргументов для запуска скрипта - сообщение об ошибке
    if(len(sys.argv)<5):
        print('No enough arguments passed!')
    else:
        #фиксируем генераторы случайных чисел чтобы обеспечить воспроизводимость результатов обучения
        os.environ['PYTHONHASHSEED']=str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        wav_file_path = sys.argv[1]   #путь к исходной аудио дорожке обучающей выборки (без клиппинга, обязательно использовать медиаконтейнер RIFF 64)
        wav_file_path1 = sys.argv[2]  #путь к аудио дорожке с клиппингом обучающей выборки (обязательно использовать медиаконтейнер RIFF 64)
        modes = [(tf.keras.layers.SimpleRNN,'relu'),(tf.keras.layers.LSTM,'tanh')]   #режимы (архитектуры рекуррентного слоя) используемые для обучения нейросети
        index = int(sys.argv[4])
        model = build_rnn_spectrogram_model(SQNC_LENGTH,modes[index][0],modes[index][1])
        model.summary()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)  #если обучение уже не дает прогресса - завершаем
        batch_size = int(sys.argv[3])  #размер пачки - целое число
        train_gen = AudioDataGenerator(wav_file_path, wav_file_path1, SQNC_LENGTH, batch_size=batch_size, cache_size=20, shuffle=True)   #генерируем обучающую выборку используя кэш
        checkpointer = ModelCheckpoint('model_checkpoint.keras',monitor='mse',verbose=1,save_best_only=True,save_weights_only=False,mode='min',save_freq='epoch')   #сохранение модели при уменьшении MSE, проверять уменьшение MSE каждую эпоху
        model.fit(train_gen,
                  verbose=1,
                  epochs=1000,
                  callbacks=[early_stopping,checkpointer])  #запустить обучение модели
