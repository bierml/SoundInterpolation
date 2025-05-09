#pragma once

#define TRUE 1 
#define FALSE 0

//структура для чтения заголовка WAV-файла
#pragma pack(push,1)   //отключить выравнивание полей структуры
struct HEADER {
	unsigned char riff[4];				//строка 'RIFF'
	unsigned int overall_size;			//общий размер файла в байтах	
	unsigned char wave[4];				//строка 'WAVE'		
	unsigned char fmt_chunk_marker[4];	//строка 'fmt ' с завершающим нулевым символом 
	unsigned int length_of_fmt;			//размер чанка 'fmt'		
	unsigned short format_type;			//формат хранения аудиоданных: 1 - PCM, остальные значения подразумевают ту или иную форму сжатия
	unsigned short channels;			//количество каналов аудио
	unsigned int sample_rate;			//частота дискретизации файла (количество семплов в секунду, герцы)
	unsigned int byterate;				//битрейт = sample_rate * channels * bits_per_sample / 8
	unsigned short block_align;			//количество байт на семпл = channgels * bits_per_sample / 8	// NumChannels * BitsPerSample/8
	unsigned short bits_per_sample;		//количество бит на семпл 		
	unsigned char data_chunk_header[4];	//заголовок чанка данных
	unsigned int data_size;				//размер чанка данных в байтах, равен num_samples * channels * bits_per_samples / 8, где num_samples - количество семплов аудио в этом чанке
};
#pragma pack(pop)

//функция чтения данных WAV-файла
//fp - файловый указатель WAV-файла (файл должен быть открыт для чтения в двоичном режиме)
//obj - указатель на структуру заголовка файла (используется для сохранения данных прочитанного заголовка файла)
//возвращаемое значение - указатель на двумерный массив прочитанных из файла семплов 
//(семплы нормируются на отрезке [-1,1], если в файле несколько каналов - каналы записываются в различные массивы) 
float** read_file_data(FILE* fp, struct HEADER* obj) {
	float** arrf = NULL;	//если по той или иной причине прочесть данные не удалось, возвращаем NULL указатель
	int read = 0;
	//действительный диапазон значений амплитуды, вычисляется на основе количества бит на семпл в заголовке файла
	long long low_limit = 0l;
	long long high_limit = 0l;
	//в заголовке WAV-файла используются 2 и 4-байтные поля, объявляем переменные для их хранения
	unsigned char buffer4[4];
	unsigned char buffer2[2];

	read = fread(obj->riff, sizeof(obj->riff), 1, fp);

	read = fread(buffer4, sizeof(buffer4), 1, fp);

	//конвертировать 4-байтное целое little-endian значение в формат big-endian
	obj->overall_size = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(obj->wave, sizeof(obj->wave), 1, fp);
	read = fread(obj->fmt_chunk_marker, sizeof(obj->fmt_chunk_marker), 1, fp);
	read = fread(buffer4, sizeof(buffer4), 1, fp);

	//конвертировать 4-байтное целое little-endian значение в формат big-endian
	obj->length_of_fmt = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer2, sizeof(buffer2), 1, fp);
	obj->format_type = buffer2[0] | (buffer2[1] << 8);
	read = fread(buffer2, sizeof(buffer2), 1, fp);
	obj->channels = buffer2[0] | (buffer2[1] << 8);
	read = fread(buffer4, sizeof(buffer4), 1, fp);
	obj->sample_rate = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer4, sizeof(buffer4), 1, fp);
	obj->byterate = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer2, sizeof(buffer2), 1, fp);
	obj->block_align = buffer2[0] |
		(buffer2[1] << 8);
	read = fread(buffer2, sizeof(buffer2), 1, fp);
	obj->bits_per_sample = buffer2[0] |
		(buffer2[1] << 8);

	read = fread(obj->data_chunk_header, sizeof(obj->data_chunk_header), 1, fp);

	read = fread(buffer4, sizeof(buffer4), 1, fp);

	obj->data_size = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	arrf = (float**)malloc(obj->channels * sizeof(float*));

	//вычислить количество семплов звуковой дорожки
	long num_samples = (8 * obj->data_size) / (obj->channels * obj->bits_per_sample);

	long size_of_each_sample = (obj->channels * obj->bits_per_sample) / 8;

	//вычислить продолжительность файла в секундах
	float duration_in_seconds = (float)obj->overall_size / obj->byterate;
	long bytes_in_each_channel = (size_of_each_sample / obj->channels);
	if (obj->format_type == 1) { //если в файле используется модуляция, отличная от PCM, чтение данных не производится
		long i = 0;
		char* data_buffer = (char*)malloc(size_of_each_sample);
		int  size_is_correct = TRUE;

		//убедиться что количество байт на семпл делится на число каналов без остатка
		if ((bytes_in_each_channel * obj->channels) != size_of_each_sample) {
			size_is_correct = FALSE;
		}

		if (size_is_correct) {

			switch (obj->bits_per_sample) {
			case 8:
				low_limit = -128;
				high_limit = 127;
				break;
			case 16:
				low_limit = -32768;
				high_limit = 32767;
				break;
			case 32:
				low_limit = -2147483648.0;
				high_limit = 2147483647;
				break;
			}
			//выделение памяти для всех семплов файла (с учетом каналов)
			for (int k = 0; k < obj->channels; k++)
				arrf[k] = (float*)malloc(num_samples * sizeof(float));
			for (i = 1; i <= num_samples; i++) {
				read = fread(data_buffer, size_of_each_sample, 1, fp);
				if (read == 1) {
					//извлечение значения семпла из прочитанных данных 
					unsigned int  xchannels = 0;
					int data_in_channel = 0;
					int offset = 0; //смещение относительно начала аудиоданных в файле
					for (xchannels = 0; xchannels < obj->channels; xchannels++) {
						//конвертировать данные из формата little-endian в формат big-endian, используя количество байт на семпл
						if (bytes_in_each_channel == 4) {
							data_in_channel = (data_buffer[offset] & 0x00ff) |
								((data_buffer[offset + 1] & 0x00ff) << 8) |
								((data_buffer[offset + 2] & 0x00ff) << 16) |
								(data_buffer[offset + 3] << 24);
						}
						else if (bytes_in_each_channel == 2) {
							data_in_channel = (data_buffer[offset] & 0x00ff) |
								(data_buffer[offset + 1] << 8);
						}
						else if (bytes_in_each_channel == 1) {
							data_in_channel = data_buffer[offset] & 0x00ff;
							data_in_channel -= 128; //в формате WAVE 8-битное аудио как правило без знака, смещаем до знакового
						}

						offset += bytes_in_each_channel;
						arrf[xchannels][i - 1] = data_in_channel / fabsf(low_limit * 1.0f);
						if (arrf[xchannels][i - 1] > 1)
						{
							arrf[xchannels][i - 1] = 1;
						}
						else if (arrf[xchannels][i - 1] < -1)
						{
							arrf[xchannels][i - 1] = -1;
						}
						//проверяем, что прочитанное значение попадает в диапазон представимых
						if (data_in_channel < low_limit || data_in_channel > high_limit)
							printf("**value out of range \n");
					}
				}
				else {
					printf("Error reading file. %d bytes\n", read);
					break;
				}
			}
			free(data_buffer);
		}
	}
	fclose(fp);
	return arrf;
}