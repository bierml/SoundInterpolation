#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <io.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wave.h"  //заголовочный файл с объявлениями, нужными для работы с WAV-файлами
#define TRUE 1 
#define FALSE 0

FILE* ptr;
FILE* fptr;
char* filename;
struct HEADER header;

#define BUFFER_SIZE 20000
#define SLOPE_LENGTH 4
//структура для хранения одного канала данных входного файла 
typedef struct {
	float* samples;
	int length;
} AudioBuffer;

//параметры устранения клиппинга во входном файле
typedef struct {
	float threshold_ratio;
	float gain_linear;
	int buffer_size;
} ClipFixParams;

//прототипы функций, используемых программой, см. описание в месте объявления функций
void clipfix_process(AudioBuffer* audio, ClipFixParams* params, float peak_level);
void processBuffer(float* buffer, int bufferLength, float threshold, int slopeLength);
void interpolate(float* buffer, int t0, int t1, int slopeLength);
float db_to_linear(float db);

int main(int argc, char** argv) {
	float** arrf = NULL;   //массив семплов исходного файла
	filename = (char*)malloc(sizeof(char) * 1024);
	if (filename == NULL) {
		printf("Error in malloc\n");
		return -1;
	}

	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {		//получить путь к текущей директории
		strcpy_s(filename, 1024,cwd);
		if (argc < 5) {
			printf("No enough arguments specified\n");
			return -2;
		}

		strcat_s(filename,1024,"\\");
		strcat_s(filename,1024,argv[1]);	//получить имя файла из командной строки
	}

	fopen_s(&ptr, filename, "rb");	//открыть входной файл
	if (ptr == NULL) {
		printf("Error opening input file\n");
		return -3;
	}

	arrf = read_file_data(ptr, &header);	//прочесть массив семплов из входного файла используя файловый указатель ptr
	if (arrf == NULL) {
		printf("Error reading input file\n");
		return -4;
	}

	//извлекаем используемые при обработке входных данных значения из заголовка входного файла
	long num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);
	long bytes_in_each_channel = header.bits_per_sample / 8;
	long limit = 32768;		//абсолютный предел шкалы представимых значений 
	switch (bytes_in_each_channel) {
		case 1:
			limit = 128;	
			break;
		case 4:
			limit = 2147483648.0;
			break;
	}
	free(filename);
	//находим пиковые значения нормированных семплов отдельно для каждого канала (входной файл не должен содержать более 4 каналов)
	float mval[4] = { 0.0,0.0,0.0,0.0 };
	for (int chan = 0; chan < header.channels; chan++)
	{
		mval[chan] = arrf[chan][0];
		for (int i = 0; i < num_samples; i++) {
			if (fabs(arrf[chan][i]) > fabs(mval[chan]))
				mval[chan] = fabs(arrf[chan][i]);
		}
	}
	AudioBuffer mybuffer[4];
	ClipFixParams params;
	//извлекаем параметры восстановления звукового файла и записываем их в структуру params
	params.buffer_size = num_samples;
	params.gain_linear = db_to_linear(strtof(argv[4], NULL));
	params.threshold_ratio = strtof(argv[3],NULL);
	//восстанавливаем каждый канал исходного аудио используя отдельный набор параметров
	for (int chan = 0; chan < header.channels; chan++)
	{
		mybuffer[chan].length = num_samples;
		mybuffer[chan].samples = arrf[chan];
		clipfix_process(&mybuffer[chan], &params, mval[chan]);	
	}
	strcat_s(cwd, 1024, "\\");
	strcat_s(cwd, 1024, argv[2]);	//формирование пути к файлу для записи полученного результата
	_unlink(cwd);
	fopen_s(&fptr, cwd, "wb");
	if (fptr == NULL) {
		printf("Error opening output file\n");
		return -5;
	}
	fwrite(&header, (size_t)sizeof(header), 1, fptr);	//записываем заголовок в результирующий файл
	int var;
	int value;
	//записать восстановленные двоичные данные в результирующий файл
	for (int j = 0; j < num_samples; j++)
	{
		for (int m = 0; m < header.channels; m++)
		{
			value = floor(arrf[m][j] * abs(limit));
			if (bytes_in_each_channel == 1)
				var = value & 0xff + 128;
			else if (bytes_in_each_channel == 2)
				var = (value & 0xff00) | (value & 0xff);
			else if (bytes_in_each_channel == 4)
				var = value;
			fwrite(&var, bytes_in_each_channel, 1, fptr);
		}
	}
	fclose(fptr);
	return 0;
}

//функция пересчета уровня громкости в дБ в относительное изменение громкости
//db - изменение громкости в децибелах
//возвращаемое значение - относительное изменение громкости соответствующее db децибел
float db_to_linear(float db) {
	return powf(10.0f, db / 20.0f);
}
//функция интерполяции кубическим эримтовым сплайном отсчетов сигнала 
//buffer - массив отсчетов интерполируемого сигнала
//t0 - индекс семпла начала отрезка, внутри которого будут интерполироваться значения
//t1 - индекс семпла конца отрезка, внутри которого будут интерполироваться значения 
//(иными словами, интерполируются семплы с индексами начиная с t0+1 до t1-1 включительно)
//dur - количество предшествующих семплов используемых для нахождения разностной производной на каждом из концов отрезка
//возвращаемое значение - нет, результаты записываются в buffer
void interpolate(float* buffer, int t0, int t1, int dur) {
	float d0 = (buffer[t0] - buffer[t0 - dur]) / dur;
	float d1 = (buffer[t1 + dur] - buffer[t1]) / dur;
	float m = (d1 + d0) / ((t1 - t0) * (t1 - t0));
	float b = (d1 / (t1 - t0)) - (m * t1);

	for (int j = t0 + 1; j < t1; j++) {
		float term1 = (t1 - j) * (buffer[t0] / (t1 - t0));
		float term2 = (j - t0) * (buffer[t1] / (t1 - t0));
		float term3 = (j - t0) * (j - t1) * (m * j + b);
		buffer[j] = term1 + term2 + term3;
	}
}
//функция восстановления множества интервалов клиппинга по заданным параметрам
//buffer - массив отсчетов интерполируемого сигнала
//buffer_length - длина массива buffer
//threshold - уровень фиксации клиппинга (уже пересчитанный с учетом относительного порога фиксации клиппинга)
//возвращаемое значение - нет, результаты записываются в buffer
void process_buffer(float* buffer, int buffer_length, float threshold) {
	int* exit_list = NULL;
	int* return_list = NULL;
	int* exit_list_ptr = NULL;
	int* return_list_ptr = NULL;
	int exit_count = 0;
	int return_count = 0;
	const int last_sample = buffer_length - SLOPE_LENGTH;  //если справа недостаточно семплов - не удастся вычислить разностную производную
	//выявить превышения уровня клиппинга
	//exit_list_ptr - массив индексов начала последовательностей клиппинга
	//return_list_ptr - массив индексов конца последовательностей клиппинга
	for (int i = SLOPE_LENGTH; i < last_sample; i++) {
		if (fabsf(buffer[i]) >= threshold) {
			if (fabsf(buffer[i - 1]) < threshold) {
				exit_list_ptr = realloc(exit_list_ptr, (exit_count + 1) * sizeof(int));
				exit_list_ptr[exit_count++] = i - 1;
			}
		}
		else {
			if (fabsf(buffer[i - 1]) >= threshold) {
				return_list_ptr = realloc(return_list_ptr, (return_count + 1) * sizeof(int));
				return_list_ptr[return_count++] = i;
			}
		}
	}
	return_list = return_list_ptr;
	exit_list = exit_list_ptr;
	//обработать граничный случай когда аудио начинается с клиппинга
	if (exit_count > 0 && return_count > 0 &&
		fabsf(buffer[SLOPE_LENGTH - 1]) >= threshold) {
		return_list++;
		return_count--;
	}

	//процесс восстановления массива областей клиппинга
	const int slope_len = SLOPE_LENGTH - 1;
	for (int i = 0; i < exit_count && i < return_count; i++) {
		interpolate(buffer, exit_list[i], return_list[i], slope_len);
	}
	if (exit_list_ptr != NULL)
		free(exit_list_ptr);
	if (return_list_ptr != NULL)
	{
		free(return_list_ptr);
		return_list_ptr = NULL;
	}
}

//процесс исправления клиппинга в буфере
//audio - буфер c восстанавливаевыми аудио данными
//params - параметры, используемые для восстановления исходных звуковых данных
//peak_level - абсолютный пик амплитуды звуковой дорожке, используется для нахождения порога фиксации клиппинга
//возвращаемое значение - нет, изменяются семплы буфера audio
void clipfix_process(AudioBuffer* audio, ClipFixParams* params, float peak_level) {
	const float threshold = params->threshold_ratio * peak_level;
	const float gain = params->gain_linear;
	const int total_samples = audio->length;

	//обработать исходный массив как множество чанков
	for (int i = 0; i < total_samples; i += params->buffer_size) {
		int chunk_size = (i + params->buffer_size > total_samples)
			? total_samples - i
			: params->buffer_size;
		process_buffer(audio->samples + i, chunk_size, threshold,BUFFER_SIZE);
	}

	//применить изменение громкости (необходимо для предотвращения клиппинга при восстановлении звука)
	for (int i = 0; i < total_samples; i++) {
		audio->samples[i] *= gain;
	}
}