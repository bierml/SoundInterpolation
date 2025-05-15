import sys
import contextlib
import wave_bwf_rf64  #модуль wave-bwf-rf64 для работы с медиаконтейнером RIFF 64
import os
from random import shuffle, seed
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  #отключить кастомные OneDNN операции

import wave
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
SQNC_LENGTH = 512   #длина входной и выходной последовательностей нейросети
FSTEP = 8  #размер шага при оконном преобразовании Фурье

#функция очищения комплексной спектрограммы звука 
#stft_complex - комплексная спектрограмма звука
#T - порог значения амплитуды спектрограммы, меньшие значения амплитуды спектрограммы будут маскироваться
#возвращаемое значение - маскированная комплексная спектрограмма звука той же формы как и входная
def mask_stft_amplitude(stft_complex: tf.Tensor, T: float) -> tf.Tensor:
    #разделение комплексной входной спектрограммы на амплитудную и фазовую составляющие, состоящие из действительных значений
    amplitude = tf.abs(stft_complex)          
    phase     = tf.math.angle(stft_complex)   

    #маскирование небольших амплитуд, если амплитуда отсчета < T, присвоить амплитуде отсчета значение 0 
    masked_amp = tf.where(amplitude < T,
                          tf.zeros_like(amplitude),
                          amplitude)

    #восстановить действительную и мнимую части сигнала
    real_part = masked_amp * tf.cos(phase)         
    imag_part = masked_amp * tf.sin(phase)         

    #воссоздать комплексный тензор аналогичного размера из действительной и мнимой частей
    return tf.complex(real_part, imag_part)

#функция для маскирования последовательностей с клиппингом и восстановленных нейросетью последовательностей
#reference - исходная последовательность с клиппингом
#restored - восстановленная нейростеью последовательность
#threshold_ratio - относительный порог фиксации клиппинга (по умолчанию - 95% абсолютного пикового значения)
#возвращаемое значение - маскированная последовательность, в которой в позициях без клиппинга использованы семплы из reference, где он есть - семплы из restored
def threshold_mask(reference,restored, threshold_ratio=0.95):
    abs_x = np.abs(reference)   
    max_val = np.max(abs_x) #находим абсолютный максимум последовательности с клиппингом
    mask = (abs_x >= (threshold_ratio * max_val)).astype(int)  #маскирование по относительному порогу: 1 - клиппинг текущего семпла обнаружен, 0 - в противном случае
    return (1-mask)*reference + restored*mask  #маска = 1 - используем семпл из restored, маска = 0 - семпл из reference

#функция для чтения WAV-файла и возвращения его семплов как NumPy массива нормированных семплов в виде значений типа float32
#file_path - путь к открываемому файлу
#возвращаемое значение - массив нормированных (значения на отрезке [-1,1]) семплов файла
def read_wav_as_float(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        #получить параметры из заголовка открываемого файла: количество каналов, количество байт на семпл, количество семплов, частота дискретизации
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        print(f"Channels: {n_channels}, Sample Width: {sample_width}, Frame Rate: {frame_rate}, Frames: {n_frames}")

        #прочесть данные исходного файла как массив значений байтов
        raw_data = wav_file.readframes(n_frames)

    #определить тип данных отдельного семпла на основании количества байт на семпл: 2 байта - np.int16, 4 байта - np.int32
    dtype = {2: np.int16, 4: np.int32}.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    #конвертировать массив сырых байтов в массив значений заданного типа 
    int_data = np.frombuffer(raw_data, dtype=dtype)

    #нормировать массив значений семплов, чтобы итоговые значения лежали в диапазоне [-1,1]
    max_val = float(2 ** (8 * sample_width - 1))  #максимальное значение семпла в знаковой PCM-модуляции когда на один семпл приходится sample_width байт 
    float_data = int_data.astype(np.float32) / max_val

    #если файл содержит несколько каналов - усреднить их значения и преобразовать результат в один канал
    if n_channels > 1:
        float_data = float_data.reshape(-1, n_channels).mean(axis=1)

    return float_data

#функция для записи нормированных значений семплов в 16-битный одноканальный WAV-файл
#samples - массив нормированных на отрезке [-1,1] семплов звукового файла
#sample_rate - частота дискретизации звуковой дорожки
#output_path - путь по которому будет записан звуковой файл
#возвращаемое значение - отсутствует
def write_float_samples_to_wav(samples, sample_rate, output_path):
    #преобразовать массив семплов в массив NumPy
    samples = np.array(samples, dtype=np.float32)

    #ограничиваем значения в массиве samples отрезком [-1,1] чтобы избежать переполнения
    samples = np.clip(samples, -1.0, 1.0)

    #конвертировать массив семплов в формат 16-битной PCM 
    int_samples = (samples * 32768).astype(np.int16)

    #записать семплы в 16-битный одноканальный WAV-файл
    with wave.open(output_path, 'wb') as wav_file:
        #установить параметры WAV-файла
        wav_file.setnchannels(1)  # Один канал
        wav_file.setsampwidth(2)  # 16-битная PCM
        wav_file.setframerate(sample_rate)  #установить частоту дискретизации равную sample_rate

        #записать семплы в аудио-файл
        wav_file.writeframes(int_samples.tobytes())

#прочитать количество семплов из заголовка WAV-файла с медиаконтейнером RIFF 64
#filename - имя файла для чтения
#возвращаемое значение - количество семплов в WAV-файле
def get_number_of_samples(filename):
    with contextlib.closing(wave_bwf_rf64.open(filename, 'rb')) as wf:
        nframes = wf.getnframes()
    return nframes
    
#функция чтения непрерывной последовательности семплов из WAV-файла с медиаконтейнером RIFF 64
#filename - путь к открываемому WAV-файлу
#start_index - индекс первого семпла читаемой последовательности (включительно - индекс первого семпла в последовательности = start_index)
#end_index - индекс последнего семпла читаемой последовательности (не включительно, т.е. реальный индекс последнего семпла = end_index-1)
#возвращаемое значение - нормированная на отрезке [-1,1] последовательность семплов файла начиная с start_index (первый семпл) и заканчивая end_index-1 (последний семпл)
def read_samples_segment(filename, start_index, end_index):
    #открыть файл в двоичном режиме для чтения
    with contextlib.closing(wave_bwf_rf64.open(filename, 'rb')) as wf:
        #извлечь из файла количество семплов в нем
        nframes = wf.getnframes()
        #индекс семпла выходит за границы массива семплов файла - выбросить исключение
        if start_index < 0 or end_index > nframes or start_index >= end_index:
            raise ValueError("Invalid start or end index.")
        
        #установить указатель в позицию начала читаемой последовательности
        wf.setpos(start_index)
        #прочитать нужное количество семплов из файла в массив frames
        frames = wf.readframes(end_index - start_index)
        
        sample_width = wf.getsampwidth()
        num_channels = wf.getnchannels()
        
        #сопоставить количеству бит на семпл тип элементов массива dtype
        if sample_width == 1:
            dtype = np.uint8  #обычно беззнаковый тип
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        #конвертировать массив сырых байтов в массив NumPy
        samples = np.frombuffer(frames, dtype=dtype)/(2**(sample_width*8-1))
        #если в файле содержался больше чем 1 канал, выделить отдельное измерение для каналов звука
        if num_channels > 1:
            samples = samples.reshape(-1, num_channels)
        return samples
    
#класс генератора обучающей выборки нейросети
#возвращает кортеж (X_seq,y_seq), где X_seq - массив (пачка) последовательностей с клиппингом (входная), y_seq - массив (пачка) последовательностей без клиппинга (выходная)
class AudioDataGenerator(Sequence):
    #функция инициализации класса генератора
    #self - ссылка на экземпляр объекта класса AudioDataGenerator
    #original_file - путь к файлу с последовательностями без клиппинга (оригинальным данным)
    #clipped_file - путь к файлу с последовательностями с клиппингом (длина в семплах должна совпадать с длиной файла без клиппинга)
    #SQNC_LENGTH - длина последовательностей возвращаемых генератором
    #batch_size - размер массива(пачки) последовательностей
    #cache_size - размер (количество пачек последовательностей) непрерывного блока данных (кэша), единоразово загружаемого в оперативную память с диска
    #shuffle - булево значение, определяющее, будет ли производиться перемешивание возвращаемой обучающей выборки
    #возвращаемое значение - нет
    def __init__(self, original_file, clipped_file, SQNC_LENGTH, batch_size = 32, cache_size = 100, shuffle=True):
        #запоминание значений инициализации в объекте генератора
        self.SQNC_LENGTH = SQNC_LENGTH
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ofname = original_file
        self.cfname = clipped_file
        self.cache_size = cache_size
        #получение количества семплов в оригинальном файле
        self.nofsamples = get_number_of_samples(original_file)
        self.step_size = SQNC_LENGTH // 2  #вычисление шага для извлечения последовательностей и генерации обучающей выборки (используем 50% наложения)
        self.indices = list(range(0, self.nofsamples - SQNC_LENGTH + 1, self.step_size))  #находим массив индексов начала последовательностей
        #сквозные индексы первой и последней последовательности кэша 
        self.startindex = -1
        self.endindex = -1
        #массивы семплов исходного файла и файла с клиппингом
        self.cache_samples_orig = []
        self.cache_samples_clip = []
        #если shuffle=True, перемешать индексы начала последовательностей в self.indices в рамках каждого кэша (self.batch_size*self.cache_size последовательностей)
        if self.shuffle:
            for i in range(int(np.ceil(len(self.indices)/(self.batch_size*self.cache_size)))):
                cache = self.indices[self.batch_size*self.cache_size*i:self.batch_size*self.cache_size*(i+1)] #извлечь self.batch_size*self.cache_size индексов
                np.random.shuffle(cache)  #перемешать
                self.indices[self.batch_size*self.cache_size*i:self.batch_size*self.cache_size*(i+1)] = cache #записать результат в массив self.indices
                
    #функция, возвращающая длину последовательности
    #self - ссылка на экземпляр объекта класса AudioDataGenerator
    #возвращаемое значение - количество пачек, приходящихся на одну эпоху
    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
        
    #функция, возвращающая следующую пачку (массив из self.batch_size последовательностей) для обучения нейросети
    #self - ссылка на экземпляр объекта класса AudioDataGenerator
    #idx - индекс пачки (число от 0 до len(obj)-1, где obj - объект генератора)
    #возвращаемое значение - пачка данных для обучения нейросети
    def __getitem__(self, idx):
        #проверить, нужно ли обновлять кэш
        if((idx//self.cache_size)*self.cache_size*self.batch_size != self.startindex):
           #обновление кэша
           cachestart_in_batch_num = (idx//self.cache_size)*self.cache_size   #номер пачки начала кэша
           self.startindex = cachestart_in_batch_num*self.batch_size    #индекс первой последовательности кэша
           #cache size in batches

           cachesz = self.cache_size*self.batch_size  #размер кэша в пачках
           self.endindex = self.startindex + cachesz - 1  #находим индекс последней последовательности кэша
           #idx - номер запрошенной пачки последовательностей 
           #self.startindex и self.endindex - индексы первой (включительно) и последней (не включительно) пачки последовательностей, хранящихся в кэше
           #пример: 3627-ая последовательность -> self.startindex = 3600, self.endindex = 3699 (для self.cache_size = 100)
           if(self.endindex > len(self.indices)-1):
               self.endindex = len(self.indices)-1
           fsqstartabsolute = self.startindex*self.step_size   #абсолютный индекс семпла начала кэша во всем файле
           lsqstartabsolute = self.endindex*self.step_size    #абсолютный индекс семпла начала последней последовательности кэша во всем файле
           lsqendabsolute = lsqstartabsolute + self.SQNC_LENGTH - 1   #абсолютный индекс семпла конца последней последовательности кэша во всем файле
           #читаем сразу весь кэш с диска в оперативную память
           self.cache_samples_orig = read_samples_segment(self.ofname,fsqstartabsolute,lsqendabsolute+1)
           self.cache_samples_clip = read_samples_segment(self.cfname,fsqstartabsolute,lsqendabsolute+1)
        #получить глобальные индексы первой и последней последовательностей для текущей пачки
        start_batch = idx * self.batch_size
        end_batch = (idx + 1) * self.batch_size
        #если пачка длиннее всего файла - усекаем индекс конца пачки
        if(end_batch>len(self.indices)-1):
            end_batch= len(self.indices)-1
        #извлечь из self.indices массив с индексами начала последовательностей пачки
        batch_indices = self.indices[start_batch : end_batch]
        #X_batch - пачка последовательностей с клиппингом
        #y_batch - пачка последовательностей без клиппинга
        X_batch = []
        y_batch = []
        #поэлементно заполняем пачку последовательностями
        for start in batch_indices:
            #извлечь две последовательности длины SQNC_LENGTH из кэша
            cachestartabs = self.startindex*self.step_size  #сквозной индекс во всем файле с которого начинается кэш
            X_seq = self.cache_samples_clip[start - cachestartabs : start - cachestartabs + self.SQNC_LENGTH]
            y_seq = self.cache_samples_orig[start - cachestartabs : start - cachestartabs + self.SQNC_LENGTH]
            X_batch.append(X_seq)
            y_batch.append(y_seq)
        #конвертировать списки последовательностей в массивы NumPy
        return np.array(X_batch), np.array(y_batch)

    def on_epoch_end(self):
        #если self.shuffle = True, перемешать куски, которыми читается кэш в эпохе, и последовательности в каждом из кэшей в  конце каждой эпохи
        if self.shuffle:
            batch_indices = list(range(int(np.ceil(len(self.indices)/(self.batch_size*self.cache_size)))))
            np.random.shuffle(batch_indices)  #перемешать последовательные куски которыми читается кэш в каждой эпохе
            for i in batch_indices:
                cache = self.indices[self.batch_size*self.cache_size*i:self.batch_size*self.cache_size*(i+1)]
                np.random.shuffle(cache)    #перемешать последовательности в кэше
                self.indices[self.batch_size*self.cache_size*i:self.batch_size*self.cache_size*(i+1)] = cache

#кастомный слой, оборачивающий весь алгоритм обработки входных последовательностей с помощью спектрограмм
class SpectrogramModelLayer(tf.keras.layers.Layer):
    #функция инициализации слоя восстановления
    #self - ссылка на экземпляр объекта класса SpectrogramModelLayer
    #sq_lngth - длина входной последовательности, обрабатываемой слоем
    #rnn_layer - тип рекуррентного слоя, используемого нейросетью (SimpleRNN или LSTM)
    #activation - активация, используевая в рекуррентном слое (с SimpleRNN лучше использовать relu, с LSTM - tanh)
    #**kwargs - словарь именованных аргументов функции
    #возвращаемое значение - нет
    def __init__(self, sq_lngth, rnn_layer=tf.keras.layers.SimpleRNN, activation='relu', **kwargs):
        super(SpectrogramModelLayer, self).__init__(**kwargs)
        #запоминание значений инициализации в объекте слоя
        self.sq_lngth = sq_lngth
        self.rnn_layer = rnn_layer
        self.activation = activation
        self.frame_step = FSTEP
        self.frame_length = FSTEP * 2  
        self.F_const = self.frame_step + 1  #количество отсчетов спектрограммы по частоте (на 1 больше из-за "нулевой" частоты - постоянной составляющей сигнала)
        self.M_const = sq_lngth // self.frame_step + 1   #количество отсчетов спетрограммы по времени (поскольку окно преобразования сдвигается с шагом self.frame_step, данное значение равно sq_lngth // self.frame_step + 1)

        #прототипы используемых рекуррентных слоев, преобразуют спектрограммы сигнала
        self.rnn = self.rnn_layer(units=sq_lngth, activation=self.activation, return_sequences=True)
        self.rnn_s = self.rnn_layer(units=sq_lngth, activation=self.activation, return_sequences=True)
        #слои, используемые непосредственно для преобразования спектрограмм сигнала
        self.rnn1 = tf.keras.layers.Bidirectional(self.rnn,merge_mode='ave')  #двунаправленный слой с усреднением результатов работы в двух направлениях
        self.dropout = tf.keras.layers.Dropout(0.25)  #слой регуляризации, выбрасывает часть нейронов при обучении
        self.rnn2 = tf.keras.layers.Bidirectional(self.rnn_s,merge_mode='ave')  #двунаправленный слой с усреднением результатов работы в двух направлениях
        self.dense = tf.keras.layers.Dense(units=self.F_const, activation='linear') #полносвязный слой 
        #слои, используемые для преобразования сигнала во временной области
        self.convout = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')  #сверточный слой с relu-активацей, 32 фильтра
        self.rnnout = self.rnn_layer(units=32, activation=self.activation, return_sequences=True)  #рекуррентный слой из 32 нейронов
        self.rnnout2 = self.rnn_layer(units=sq_lngth//2, activation=self.activation, return_sequences=True)  #рекуррентный слой из sq_lngth//2 нейронов
        self.denseout = tf.keras.layers.Dense(units=sq_lngth//2, activation='linear') #полносвязный слой из sq_lngth//2 нейронов
        self.convout2 = tf.keras.layers.Conv1D(filters=1, kernel_size=3, activation='linear', padding='same')  #сверточный слой для устранения искажений во временной области
    #функция восстановления сигнала с помощью разработанной нейросети
    #self - ссылка на экземпляр объекта класса SpectrogramModelLayer
    #inputs - восстанавливаемая последовательность во временной области (или пачка из таких последовательностей, т.е. форма входа (None,sq_lngth))
    #возвращаемое значение - последовательность или пачка последовательностей такой же формы как inputs
    def call(self, inputs):
        #форма входной последовательности (batch, self.sq_lngth)
        #преобразуем сигнал в спектрограмму с помощью оконного преобразования Фурье
        #self.frame_length = FSTEP * 2, self.frame_step = FSTEP, степень наложения - 50%
        #mag - амплитудная составляющая комплексных отсчетов спектрограммы
        #phase - фазовая составляющая комплексных отсчетов спектрограммы 
        #при восстановлении преобразуем только амплитудную составляющую спектрограммы, фазовую оставляем неизменной
        mag, phase = tf.keras.ops.stft(inputs,self.frame_length,self.frame_step,self.frame_length)
        batch_size = tf.shape(mag)[0]   #извлекаем количество последовательностей в пачке
        #используем последовательные рекуррентные слои для преобразования спектрограммы, вход (self.M_const,self.F_const)->(self.M_const,self.sq_lngth)
        #рекуррентные слои идеально подходят для обработки изменяющихся срезов спектрограммы по времени за счет наличия "состояния"
        x = self.rnn1(mag) 
        x = self.dropout(x)  #отбрасываем часть нейронов при градиентном спуске чтобы избежать переобучения
        x = self.rnn2(x)
        x = self.dense(x)   #возвращаем спектрограмму в исходную форму, устраняя избыточность (self.M_const,self.sq_lngth)->(self.M_const,self.F_const)
        x = mag + x  #прибавляем получившийся результат к спектрограмме входной последовательности
        #восстановить сигнал во временной области с помощью обратного оконного преобразования Фурье, для правильной инверсии используем те же значения параметров что и при прямом преобразовании
        reconstructed = tf.keras.ops.istft([x,phase],self.frame_length,self.frame_step,self.frame_length)

        #преобразование последовательности во временной области
        reconstructed = tf.reshape(reconstructed, [batch_size, self.sq_lngth, 1])   #добавляем дополнительное измерение сигнала
        #сверточный слой, преобразует форму сигнала (batch_size, self.sq_lngth, 1)->(batch, self.sq_lngth, 32), т.е. добавляется избыточность
        rec_proc1 = self.convout(reconstructed)  
        #преобразуем входные данные через рекуррентные слои 
        rec_proc = self.rnnout(rec_proc1)    #(batch, self.sq_lngth, 32)->(batch, self.sq_lngth, 32), степень избыточности та же
        rec_proc = self.dropout(rec_proc)     #отбрасываем часть нейронов при градиентном спуске чтобы избежать переобучения
        rec_proc = self.rnnout2(rec_proc)  #(batch, self.sq_lngth, 32)->(batch, sq_lngth, self.sq_lngth//2), степень избыточности увеличена
        rec_proc = self.denseout(rec_proc)   #полносвязный слой, (batch, self.sq_lngth, self.sq_lngth//2)->(batch, self.sq_lngth, self.sq_lngth//2)
        rec_proc = self.convout2(rec_proc)        #убираем избыточность через свертку->(batch, self.sq_lngth, self.sq_lngth//2)->(batch, sq_lngth, 1)
        rec_proc = tf.squeeze(rec_proc, axis=-1)  #убираем дополнительное измерение (batch, sq_lngth, 1)->(batch, sq_lngth)

        #обратная связь, складываем получившиеся восстановленные пики во временной области с входным сигналом
        return inputs + rec_proc
    #функция, возвращающая словарь с параметрами слоя: нужна для правильного сохранения модели в файл и загрузки модели из файла
    #self - ссылка на экземпляр объекта класса SpectrogramModelLayer
    #возвращаемое значение - словарь, содержащий пары (ключ,значение) вида (название_параметра, значение)
    def get_config(self):
        config = super(SpectrogramModelLayer, self).get_config()
        #возвращать дополнительно к основным параметрам слоя: длину последовательности, тип рекуррентного слоя, активацию
        config.update({
            "sq_lngth": self.sq_lngth,
            "rnn_layer": self.rnn_layer.__name__,
            "activation": self.activation
        })
        return config
    #функция, создающая слой на основании его конфигурации
    #cls - класс, объект которого инстанцируется (в данном случае, SpectrogramModelLayer)
    #config - словарь с параметрами слоя, содержащий пары (ключ,значение) вида (название_параметра, значение)
    #возвращаемое значение - объект класса cls, инициализированный параметрами из словаря config
    @classmethod
    def from_config(cls, config):
        #преобразуем строковое название слоя в класс слоя
        rnn_layer_name = config.pop("rnn_layer")
        if rnn_layer_name == "LSTM":
            config["rnn_layer"] = tf.keras.layers.LSTM
        elif rnn_layer_name == "SimpleRNN":
            config["rnn_layer"] = tf.keras.layers.SimpleRNN
        else:
            raise ValueError(f"Unsupported rnn_layer: {rnn_layer_name}") #если такой строке никакой слой не соответствует, возвращаем ошибку
        #инстанциируем объект данного класса
        return cls(**config)

#функция, реализующая модель на основании заданных параметров
#sq_lngth - длина последовательности
#layer - рекуррентный слой, используемый для инициализации модели
#activation - функция активации, применяемая сразу после рекуррентных слоев модели
#возвращаемое значение - объект модели для обучения с заданными параметрами
def build_rnn_spectrogram_model(sq_lngth,layer,activation):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(sq_lngth,)),
        SpectrogramModelLayer(sq_lngth=sq_lngth,rnn_layer=layer,activation=activation)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001,weight_decay=0.01),
        loss='mse',
        metrics=['mse']
    )
    return model
    