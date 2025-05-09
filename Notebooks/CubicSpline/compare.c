#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <io.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "wave.h"  //заголовочный файл с объявлениями, нужными для работы с WAV-файлами

//храним комплексные числа в виде действительной и мнимой частей в структуре
typedef struct {
	double Re;
	double Im;
} complex;

//функция проверки, является ли x степенью двойки
//x - целое число
//возвращаемое значение - булево, true, если x - степень двойки, false - в противном случае
bool IsPowerOfTwo(long x)
{
	return (x != 0) && ((x & (x - 1)) == 0);
}

//быстрое преобразование Фурье для длин массива, выражаемых степенью двойки
//v - комплексные отсчеты исходного сигнала
//n - количество комплексных отсчетов исходного сигнала
//возвращаемое значение - флаг ошибки при выполнении БПФ (неверная длина входного массива - 1, иначе - 0)
int fft(complex* v, int n)
{
	//проверяем, является ли длина входного массива степенью двойки
	if (!IsPowerOfTwo(n))
		return 1;
	//выделяем память для результатов ДПФ (результат - массив комплексных значений равный по длине входному массиву)
	complex* tmp = (complex*)malloc(n * sizeof(complex));
	if (n > 1) {			//если массив не содержит хотя бы 2 элементов - искомое БПФ уже найдено, ничего не делаем
		int k, m;    
		complex z, w, * vo, * ve;
		ve = tmp; vo = tmp + n / 2;  //ve - указатель на начало первой половины tmp, vo - на начало второй половины tmp
        //кладем четные отсчеты v в ve, нечетные в vo
		for (k = 0; k < n / 2; k++) {
			ve[k] = v[2 * k];
			vo[k] = v[2 * k + 1];
		}
		fft(ve, n / 2);		//БПФ для первой половины массива tmp (четных элементов) 
		fft(vo, n / 2);		//БПФ для второй половины массива tmp (нечетных элементов) 
		//объединение БПФ для n/2 нечетных и n/2 четных отсчетов в БПФ для всей длины n
		//осуществляется по формулам:
		//X_m=E_m+e^{-2\pi{i}\frac{m}{N}}\cdot O_m
		//X_{m+\frac{N}{2}}=E_m-e^{-2\pi{i}\frac{m}{N}}\cdot O_m
		//здесь: O_m - ДПФ для элемента x_{2m+1}
		//E_m - ДПФ для элемента x_{2m}
		//m - целое число, такое что 0 <= m < N/2
		for (m = 0; m < n / 2; m++) {
			//находим e^{-2\pi{i}\frac{m}{N}} в виде действительной и мнимой частей, далее обозначим это значение как w
			w.Re = cos(2 * M_PI * m / (double)n);
			w.Im = sin(2 * M_PI * m / (double)n);
			z.Re = w.Re * vo[m].Re - w.Im * vo[m].Im;	//действительная часть w \cdot O_m
			z.Im = w.Re * vo[m].Im + w.Im * vo[m].Re;	//мнимая часть w \cdot O_m
			v[m].Re = ve[m].Re + z.Re;		//действительная часть E_m+w \cdot O_m
			v[m].Im = ve[m].Im + z.Im;		//мнимая часть E_m+w \cdot O_m
			v[m + n / 2].Re = ve[m].Re - z.Re;	//действительная часть E_m - w \cdot O_m
			v[m + n / 2].Im = ve[m].Im - z.Im;	//мнимая часть E_m - w \cdot O_m
		}
	}
	free(tmp);
	return 0;
}
//файловые указатели, строки имен и структуры заголовков для двух сравниваемых файлов
FILE* ptr1;
FILE* ptr2;
char* filename1;
char* filename2;
struct HEADER header1;
struct HEADER header2;
//функция сравнения заголовков обрабатываемых программой файлов
//hd1 - заголовок первого обрабатываемого файла
//hd2 - заголовок второго обрабатываемого файла
//возвращаемое значение - булево, true - заголовки совпадают, false - заголовки различаются
bool compare_headers(struct HEADER hd1, struct HEADER hd2) {
	//для числовых полей сравнение через ==, для строк - проверяем равенство нулю результата strcmp
	bool res = (strcmp(hd1.riff, hd2.riff) == 0) && (hd1.overall_size == hd2.overall_size) && \
		(strcmp(hd1.wave, hd2.wave) == 0) && (strcmp(hd1.fmt_chunk_marker, hd2.fmt_chunk_marker) == 0) && \
		(hd1.length_of_fmt == hd2.length_of_fmt) && (hd1.format_type == hd2.format_type) && (hd1.channels == hd2.channels) && \
		(hd1.sample_rate == hd2.sample_rate) && (hd1.byterate == hd2.byterate) && (hd1.block_align == hd2.block_align) && \
		(hd1.bits_per_sample == hd2.bits_per_sample) && (strcmp(hd1.data_chunk_header, hd2.data_chunk_header) == 0) && (hd1.data_size == hd2.data_size);
	return res;
}

int main(int argc, char** argv) {
	//массивы для хранения семплов прочитанных файлов
	float** samples1 = NULL;  
	float** samples2 = NULL;

	filename1 = (char*)malloc(sizeof(char) * 1024);
	filename2 = (char*)malloc(sizeof(char) * 1024);
	if (filename1 == NULL || filename2 == NULL) {
		printf("Error in malloc\n");
		exit(1);
	}
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {  	//получить путь к текущему рабочему каталогу

		strcpy_s(filename1, 1024, cwd);
		strcpy_s(filename2, 1024, cwd);
		if (argc < 3) {
			printf("No wave file specified\n");
			return -1;
		}
		//получить имена файлов из командной строки
		strcat_s(filename1, 1024, "\\");
		strcat_s(filename1, 1024, argv[1]);
		strcat_s(filename2, 1024, "\\");
		strcat_s(filename2, 1024, argv[2]);
		printf("Original file: %s\n", filename1);
		printf("Clipped file: %s\n", filename2);
	}
	//открыть исходные файлы в двоичном режиме чтения
	fopen_s(&ptr1, filename1, "rb");
	fopen_s(&ptr2, filename2, "rb");
	if (ptr1 == NULL || ptr2 == NULL) {
		printf("Error opening file\n");
		exit(1);
	}
	//samples1 - массив семплов оригинального файла, samples2 - искаженного
	samples1 = read_file_data(ptr1, &header1);
	samples2 = read_file_data(ptr2, &header2);
	if (!compare_headers(header1, header2)) {   //если заголовки не совпадают, сообщить о проблеме и вывести дамп заголовков
		printf("Given files are not in the same format!\n");
		printf("Header 1:\n\t");
		unsigned char* tmp = &header1;
		for (int i = 0; i < sizeof(header1); i++) {
			printf("%02X ", tmp[i]);
		}
		printf("\n");
		printf("Header 2:\n\t");
		tmp = &header2;
		for (int i = 0; i < sizeof(header2); i++) {
			printf("%02X ", tmp[i]);
		}
		printf("\n");
		exit(1);
	}
	//значения MSE и максимальные по модулю значения семплов в файлах (отдельные для каждого канала)
	double* mse = (double*)malloc(header1.channels * sizeof(double));
	double* mse1 = (double*)malloc(header1.channels * sizeof(double));
	double* maxs = (double*)malloc(header1.channels * sizeof(double));
	double* mse2 = (double*)malloc(header1.channels * sizeof(double));
	if (mse == NULL || mse1 == NULL || mse2 == NULL || maxs == NULL) {
		printf("Error in malloc\n");
		exit(1);
	}
	long num_samples = (8 * header1.data_size) / (header1.channels * header1.bits_per_sample);
	long real_num_samples = pow(2, ceil(log(num_samples) / log(2)));
	double norm = 0.0;
	double alpha = 0.0;
	//цикл нахождения заданного набора 5 показателей для каждого из каналов звуковых файлов
	for (int channel = 0; channel < header1.channels; channel++)
	{
		//нахождение коэффициента приведения масштаба alpha
		for (int i = 0; i < num_samples; i++)
			norm += samples1[channel][i] * samples1[channel][i];
		for (int i = 0; i < num_samples; i++)
			alpha += samples1[channel][i] * samples2[channel][i] / norm;
		mse[channel] = 0.0f;
		mse1[channel] = 0.0f;
		mse2[channel] = 0.0f;
		maxs[channel] = samples1[channel][0];
		//БПФ будет применяться к массивам семплов исходных файлов, дополненных нулевыми отсчетами до длины, выражаемой степенью двойки
		complex* FFT1 = (complex*)calloc(1,real_num_samples * sizeof(complex));
		complex* FFT2 = (complex*)calloc(1,real_num_samples * sizeof(complex));
		if (FFT1 == NULL || FFT2 == NULL)
		{
			printf("Error in malloc\n");
			exit(1);
		}
		for (int i = 0; i < num_samples; i++)
		{
			FFT1[i].Re = (double)samples1[channel][i];
			FFT2[i].Re = (double)samples2[channel][i];
			if (fabs(samples1[channel][i]) > fabs(maxs[channel]))
				maxs[channel] = samples1[channel][i];	//максимальное значение семпла в каждом канале исходного файла
			mse1[channel] += pow(samples1[channel][i], 2) / num_samples;  //среднеквадратичное значение семпла исходного файла
			mse[channel] += pow((samples1[channel][i] - samples2[channel][i]), 2) / num_samples;  //MSE двух файлов
			mse2[channel] += pow((alpha*samples1[channel][i] - samples2[channel][i]), 2) / num_samples;	//MSE двух файлов с поправкой
		}
		//находим БПФ двух исходных файлов, результаты используются для нахождения LSD
		fft(FFT1, real_num_samples);
		fft(FFT2, real_num_samples);
		double LSD = 0.0;
		//нахождение значения LSD^2
		for (int i = 0; i < real_num_samples; i++)
			LSD += 100/(double)real_num_samples * pow(log10((FFT1[i].Re * FFT1[i].Re + FFT1[i].Im * FFT1[i].Im) / (FFT2[i].Re * FFT2[i].Re + FFT2[i].Im * FFT2[i].Im)), 2);
		//нахождение SNR,SI-SNR,PSNR,LSD
		LSD = sqrt(LSD);
		double SDR = 10 * log10f(mse1[channel] / mse[channel]);
		double SISDR = 10 * log10f(alpha*alpha*mse1[channel] / mse2[channel]);
		double PSNR = 10 * log10f(maxs[channel] * maxs[channel] / mse[channel]);
		//вывод найденных значений для каждого канала с фиксированной точностью вывода
		printf("------------------Channel %d------------------\n",channel+1);
		printf("MSE = %12.9lf\n", mse[channel]);
		printf("SNR = %12.9lf dB\n", SDR);
		printf("SI-SNR = %12.9lf dB\n", SISDR);
		printf("PSNR = %12.9lf dB\n", PSNR);
		printf("LSD = %12.9lf dB\n", LSD);
		free(FFT1);
		free(FFT2);
	}
	//очистка выделенных из кучи данных 
	for (int k = 0; k < header1.channels; k++)
		free(samples1[k]);
	free(samples1);
	for (int k = 0; k < header1.channels; k++)
		free(samples2[k]);
	free(samples2);
	free(mse);
	free(mse1);
	free(mse2);
	free(maxs);
	free(filename1);
	free(filename2);
}
