from base import *
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
from tensorflow.keras.utils import custom_object_scope
if __name__ == '__main__':
    #передано меньше 4 аргументов - выводим сообщение об ошибке
    if(len(sys.argv)<5):
        print('No enough arguments passed!')
    else:
        model_for_load_path = sys.argv[1] #путь к загружаемому файлу обученной модели
        file_for_restoration_path = sys.argv[2] #путь к восстанавливаемому WAV-файлу
        output_path = sys.argv[3]  #путь сохранения восстановленного WAV-файла
        alpha = float(sys.argv[4]) #относительный порог фиксации клиппинга - действительное число на интервале (0,1)
        
        #класс, используемый для корретной загрузки входного слоя (отбрасывание измерения, отвечающего за размер пачки) 
        class CustomInputLayer(tf.keras.layers.InputLayer):
            @classmethod
            def from_config(cls, config):
                #если в InputLayer в сохраненном файле есть измерение с размером пачки, удалить его
                if 'batch_shape' in config:
                    config.pop('batch_shape')
                return super(CustomInputLayer, cls).from_config(config)
        
        #загрузить частоту дискретизации из заголовка восстанавливаемого файла
        with wave.open(file_for_restoration_path, 'rb') as wav_file:
            fs = wav_file.getframerate()
        
        #загрузка модели из файла с обученной моделью с использованием объявленных слоев и функций из base
        model = tf.keras.models.load_model(
            model_for_load_path,
            custom_objects={
                'SpectrogramModelLayer': SpectrogramModelLayer,
                'InputLayer': CustomInputLayer,
                'DTypePolicy': Policy,
                'mse_shortened': mse_shortened
            }
        )

        samples_input_file = read_wav_as_float(file_for_restoration_path) #чтение семплов из восстанавливаемого звукого файла
        restored_samples_overlap = []
        overlap_input_sequences = []
        step_size = SQNC_LENGTH // 2 #наложение между соседними последовательностями - 50%
        j = 0
        #нахождение максимума и минимума значений семплов восстанавливаемого файла
        maxv = np.max(np.array(samples_input_file))
        minv = np.min(np.array(samples_input_file))
        #разбиение массива семплов восстанавливаемого файла на последовательности из SQNC_LENGTH семплов с наложением в 50%
        while j < len(samples_input_file):
            #если индекс конца извлекаемой последовательности в пределах массива семплов - добавляем
            if(j+SQNC_LENGTH < len(samples_input_file)):
                overlap_input_sequences.append(samples_input_file[j:j+SQNC_LENGTH])
            else: #иначе - завершаем цикл
                break
            j += step_size
        overlap_input_sequences = np.array(overlap_input_sequences)
        nn_restored = model.predict_on_batch(overlap_input_sequences) #восстановление массива последовательностей с помощью загруженной нейросети
        i = 0
        #получение окончательного результата восстановления - где нет клиппинга берем семпды из восстанавливаемого файла, где есть - результаты работы нейросети
        for sqnc in overlap_input_sequences:
            #если в последовательности есть клиппинг - вставляем маскированные результаты восстановления нейросети
            if(max(sqnc)>(maxv*alpha) or min(sqnc)<(minv*alpha)):
                #извлекаем семплы из середины последовательностей исходных и восстановленных семплов
                restored = nn_restored[i][SQNC_LENGTH//4:(SQNC_LENGTH*3)//4]
                reference = np.array(sqnc[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4])
                #объединяем результаты в итоговую последовательность, используя относительный порог клиппинга alpha
                restored_samples_overlap.append(threshold_mask(reference, restored, alpha))
            #в противном случае - вставляем последовательность из исходного файла
            else:
                restored_samples_overlap.append(np.array(sqnc[SQNC_LENGTH//4:(SQNC_LENGTH*3)//4]))
            i += 1
        restored_samples_overlap = np.array(restored_samples_overlap).flatten()
        #вычисляем количество семплов в конце, не покрытых последней последовательностью 
        add = len(samples_input_file) - SQNC_LENGTH//4 - restored_samples_overlap.shape[0]
        #дополняем массив SQNC_LENGTH//4 семплами из исходного файла в начале (были пропущены по причине того, что мы вставляем семплы из середины каждой последовательности)
        restored_samples_overlap = np.append(np.array(samples_input_file[0:SQNC_LENGTH//4]),restored_samples_overlap)
        #дополнение массива add семплами на конце
        restored_samples_overlap = np.append(restored_samples_overlap,np.array(samples_input_file[-add:]))
        #запись результатов восстановления по пути output_path
        write_float_samples_to_wav(restored_samples_overlap, fs, output_path)  
        print(f"WAV file written to {output_path}")



