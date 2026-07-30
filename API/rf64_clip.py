import struct
import numpy as np
import random
import sys

#функция пересчета уровня клиппинга в децибелах в относительный порог отсечения
#dbvalue - отрицательный уровень клиппинга в децибелах 
#возвращаемое значение - относительный (т.е. значение лежит на полуотрезке (0,1] для положительного dbvalue) порог клиппинга
def db_to_multiplicator(dbvalue):
    return 10 ** (-dbvalue / 20.0)
#функция чтения заголовка чанка файла
#f - дескриптор открытого файла
#возвращаемое значение - кортеж из двух значений (tag,size), где tag - название чанка, size - его размер в байтах
def read_chunk_header(f):
    hdr = f.read(8)
    if len(hdr) < 8:
        return None, None  #если прочитано меньше 8 байт - чтение не удалось, возвращаем None
    #парсим прочитанные байты в значения переменных
    #'<4sL' - формат данных: < - порядок байтов little-endian, 4s - первые 4 прочитанных байта это строка из 4 символов, L - последние 4 прочитанных байта это длинное целое значение без знака
    tag, size = struct.unpack('<4sL', hdr)  
    return tag, size
if __name__ == '__main__':
    #недостаточно аргументов для запуска скрипта - сообщение об ошибке
    if(len(sys.argv)<5):
        print('No enough arguments passed!')
    else:
        #пользовательские параметры
        source_file_path = sys.argv[1]   #имя входного файла для внесения клиппинга
        destination_file_path = sys.argv[2]  #имя выходного файла, в который будет записана дорожка после внесения клиппинга
        clip_thrshd_db_min = float(sys.argv[3])   #минимальный уровень клиппинга в децибелах
        clip_thrshd_db_max = float(sys.argv[4])   #максимальный уровень клиппинга в децибелах
        chunk_size = 44100   #длина блока, к которому применяется один уровень клиппинга
        
        #открыть входной файл и распарсить заголовок
        with open(source_file_path, 'rb') as fin:
            #прочесть первые 12 байт: подпись, размер файла, строку "WAVE"
            riff_header = fin.read(12)
            #не удалось прочесть заголовок - выбросить исключение
            if len(riff_header) < 12:
                raise ValueError("File too short")
            #прочесть подпись файла, если медиаконтейнер не RIFF 64 - выбросить исключение
            signature = riff_header[:4]
            if signature != b'RF64':
                raise ValueError("Not an RF64 file (signature = {})".format(signature))
            
            header_bytes = riff_header   #мы соберем байты заголовка в этой переменной
            data_chunk_found = False
            data_chunk_size = None
            extended_data_size = None  #в этой переменной будет храниться количество байт аудиоданных (считывается из ds64 чанка)
            
            #считывать чанки из файла пока не найдем чанк 'data'
            while not data_chunk_found:
                tag, size = read_chunk_header(fin)
                if tag is None:
                    raise ValueError("Reached end of file without finding data chunk")
                header_bytes += struct.pack('<4sL', tag, size)
                if tag == b'ds64':
                    ds64_data = fin.read(size)
                    header_bytes += ds64_data
                    #распаковать первые 24 байта: поля riffSize (размер блока RF64), dataSize (размер чанка 'data'), sampleCount (фактическое количество семплов); все поля 64-битные little-endian
                    if size >= 24:
                        riffSize_ext, data_size_ext, sample_count_ext = struct.unpack('<QQQ', ds64_data[:24])
                        extended_data_size = data_size_ext
                elif tag == b'fmt ':
                    fmt_data = fin.read(size)
                    header_bytes += fmt_data
                    sampwidth = struct.unpack('<H',fmt_data[14:16])[0]//8   #число байт на семпл
                    print("Sample width:",sampwidth)
                elif tag == b'data':
                    data_chunk_found = True
                    data_chunk_size = size
                    #достигли чанка 'data' - завершаем цикл
                else:
                    #для любого другого чанка, скопируем его целиком
                    chunk_data = fin.read(size)
                    header_bytes += chunk_data

            if data_chunk_size is None:
                raise ValueError("Data chunk not found in source file")

            #сопоставить количеству бит на семпл тип элементов массива dtype
            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            #если 32-битный размер в чанке 'data' равен 0xFFFFFFFF в 16-ричной системе, использовать 64-битный размер из чанка ds64
            if data_chunk_size == 0xFFFFFFFF and extended_data_size is not None:
                data_chunk_size = extended_data_size

            print("Header length:", len(header_bytes))
            print("Data chunk size (bytes):", data_chunk_size)

            #в этот момент, fin располагается в начале чанка data
            with open(destination_file_path, 'wb') as fout:   #открыть файл результата в двоичном режиме записи
                #записать заголовок в точности как он был прочтен
                fout.write(header_bytes)
                #обработать чанк данных по частям
                bytes_remaining = data_chunk_size

                #при PCM со знаком, максимальное по модулю значение семпла равно 2^(sample_width*8-1)
                full_scale = 2**(sampwidth*8-1)

                while bytes_remaining > 0:
                    dbval = random.uniform(clip_thrshd_db_min,clip_thrshd_db_max)
                    clip_threshold = int(db_to_multiplicator(dbval) * full_scale)
                    #принудительное устанавливаем количество читаемых байт кратное числу байт на семпл
                    to_read = min(chunk_size * sampwidth, (bytes_remaining // sampwidth) * sampwidth)
                    raw_chunk = fin.read(to_read)
                    if not raw_chunk:
                        break
                    #конвертировать массив сырых байтов в массив NumPy
                    samples = np.frombuffer(raw_chunk, dtype=dtype)
                    #произвести клиппинг прочитанных данных
                    clipped = np.clip(samples, -clip_threshold, clip_threshold)
                    #записать обработанный чанк в выходной файл
                    fout.write(clipped.tobytes())
                    bytes_remaining -= len(raw_chunk)
        print("Processing complete. Processed file saved as:", destination_file_path)