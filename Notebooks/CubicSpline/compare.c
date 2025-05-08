#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <io.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include "wave.h"
#define TRUE 1 
#define FALSE 0

typedef struct {
	double Re;
	double Im;
} complex;


bool IsPowerOfTwo(long x)
{
	return (x != 0) && ((x & (x - 1)) == 0);
}

/* Print a vector of complexes as ordered pairs. */
static void print_vector(const char* title, complex* x, int n)
{
	int i;
	printf("%s (dim=%d):", title, n);
	for (i = 0; i < n; i++) printf(" %5.2f,%5.2f ", x[i].Re, x[i].Im);
	putchar('\n');
	return;
}

void fft(complex* v, int n, int inv)
{
	int sign;
	complex* tmp = (complex*)malloc(n * sizeof(complex));
	if (n > 1) {			/* otherwise, do nothing and return */
		int k, m;    complex z, w, * vo, * ve;
		ve = tmp; vo = tmp + n / 2;
		for (k = 0; k < n / 2; k++) {
			ve[k] = v[2 * k];
			vo[k] = v[2 * k + 1];
		}
		fft(ve, n / 2, inv);		/* FFT on even-indexed elements of v[] */
		fft(vo, n / 2, inv);		/* FFT on odd-indexed elements of v[] */
		if (inv == 1)
			sign = 1;
		else
			sign = -1;
		for (m = 0; m < n / 2; m++) {
			w.Re = cos(2 * M_PI * m / (double)n);
			w.Im = sign * sin(2 * M_PI * m / (double)n);
			z.Re = w.Re * vo[m].Re - w.Im * vo[m].Im;	/* Re(w*vo[m]) */
			z.Im = w.Re * vo[m].Im + w.Im * vo[m].Re;	/* Im(w*vo[m]) */
			v[m].Re = ve[m].Re + z.Re;
			v[m].Im = ve[m].Im + z.Im;
			v[m + n / 2].Re = ve[m].Re - z.Re;
			v[m + n / 2].Im = ve[m].Im - z.Im;
		}
	}
	free(tmp);
	return;
}

//n is power of 2
int fft_(complex* v, int n, int inv)
{
	if (!IsPowerOfTwo(n))
		return 1;
	fft(v, n, inv);
	if (inv == 1) {
		for (int i = 0; i < n; i++)
		{
			v[i].Re = v[i].Re / n;
			v[i].Im = v[i].Im / n;
		}
	}
	return 0;
}

FILE* ptr1;
FILE* ptr2;
char* filename1;
char* filename2;
struct HEADER header1;
struct HEADER header2;
bool compare_headers(struct HEADER hd1, struct HEADER hd2) {
	bool res = (strcmp(hd1.riff, hd2.riff) == 0) && (hd1.overall_size == hd2.overall_size) && \
		(strcmp(hd1.wave, hd2.wave) == 0) && (strcmp(hd1.fmt_chunk_marker, hd2.fmt_chunk_marker) == 0) && \
		(hd1.length_of_fmt == hd2.length_of_fmt) && (hd1.format_type == hd2.format_type) && (hd1.channels == hd2.channels) && \
		(hd1.sample_rate == hd2.sample_rate) && (hd1.byterate == hd2.byterate) && (hd1.block_align == hd2.block_align) && \
		(hd1.bits_per_sample == hd2.bits_per_sample) && (strcmp(hd1.data_chunk_header, hd2.data_chunk_header) == 0) && (hd1.data_size == hd2.data_size);
	return res;
}
float** read_file_data(FILE* fp, struct HEADER* obj) {
	float** arrf = NULL;
	int read = 0;
	// the valid amplitude range for values based on the bits per sample
	long long low_limit = 0l;
	long long high_limit = 0l;
	// read header parts
	unsigned char buffer4[4];
	unsigned char buffer2[2];

	read = fread(obj->riff, sizeof(obj->riff), 1, fp);

	read = fread(buffer4, sizeof(buffer4), 1, fp);

	// convert little endian to big endian 4 byte int
	obj->overall_size = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(obj->wave, sizeof(obj->wave), 1, fp);
	read = fread(obj->fmt_chunk_marker, sizeof(obj->fmt_chunk_marker), 1, fp);
	read = fread(buffer4, sizeof(buffer4), 1, fp);

	// convert little endian to big endian 4 byte integer
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
	// calculate no.of samples
	long num_samples = (8 * obj->data_size) / (obj->channels * obj->bits_per_sample);

	long size_of_each_sample = (obj->channels * obj->bits_per_sample) / 8;

	// calculate duration of file
	float duration_in_seconds = (float)obj->overall_size / obj->byterate;
	long bytes_in_each_channel = (size_of_each_sample / obj->channels);
	if (obj->format_type == 1) { // PCM
		long i = 0;
		char* data_buffer = (char*)malloc(size_of_each_sample);
		int  size_is_correct = TRUE;

		// make sure that the bytes-per-sample is completely divisible by num.of channels
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
			//we need to add full allocation of arrf here
			for (int k = 0; k < obj->channels; k++)
				arrf[k] = (float*)malloc(num_samples * sizeof(float));
			for (i = 1; i <= num_samples; i++) {
				read = fread(data_buffer, size_of_each_sample, 1, fp);
				if (read == 1) {
					// dump the data read
					unsigned int  xchannels = 0;
					int data_in_channel = 0;
					int offset = 0; // move the offset for every iteration in the loop below
					for (xchannels = 0; xchannels < obj->channels; xchannels++) {
						// convert data from little endian to big endian based on bytes in each channel sample
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
							data_in_channel -= 128; //in wave, 8-bit are unsigned, so shifting to signed
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
						// check if value was in range
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


int main(int argc, char** argv) {
	float** samples1 = NULL;
	float** samples2 = NULL;

	filename1 = (char*)malloc(sizeof(char) * 1024);
	filename2 = (char*)malloc(sizeof(char) * 1024);
	if (filename1 == NULL) {
		printf("Error in malloc\n");
		exit(1);
	}
	// get file path
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {

		strcpy_s(filename1, 1024, cwd);
		strcpy_s(filename2, 1024, cwd);
		// get filename from command line
		if (argc < 3) {
			printf("No wave file specified\n");
			return -1;
		}

		strcat_s(filename1, 1024, "\\");
		strcat_s(filename1, 1024, argv[1]);
		strcat_s(filename2, 1024, "\\");
		strcat_s(filename2, 1024, argv[2]);
		printf("Original file: %s\n", filename1);
		printf("Clipped file: %s\n", filename2);
	}
	// open files
	fopen_s(&ptr1, filename1, "rb");
	fopen_s(&ptr2, filename2, "rb");
	if (ptr1 == NULL || ptr2 == NULL) {
		printf("Error opening file\n");
		exit(1);
	}
	//suppose samples1 is original file data
	samples1 = read_file_data(ptr1, &header1);
	samples2 = read_file_data(ptr2, &header2);
	if (!compare_headers(header1, header2)) {
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
	for (int channel = 0; channel < header1.channels; channel++)
	{
		for (int i = 0; i < num_samples; i++)
			norm += samples1[channel][i] * samples1[channel][i];
		for (int i = 0; i < num_samples; i++)
			alpha += samples1[channel][i] * samples2[channel][i] / norm;
		mse[channel] = 0.0f;
		mse1[channel] = 0.0f;
		mse2[channel] = 0.0f;
		maxs[channel] = samples1[channel][0];
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
				maxs[channel] = samples1[channel][i];
			mse1[channel] += pow(samples1[channel][i], 2) / num_samples;
			mse[channel] += pow((samples1[channel][i] - samples2[channel][i]), 2) / num_samples;
			mse2[channel] += pow((alpha*samples1[channel][i] - samples2[channel][i]), 2) / num_samples;
		}
		fft_(FFT1, real_num_samples, 0);
		fft_(FFT2, real_num_samples, 0);
		double LSD = 0.0;
		for (int i = 0; i < real_num_samples; i++)
			LSD += 100/(double)real_num_samples * pow(log10((FFT1[i].Re * FFT1[i].Re + FFT1[i].Im * FFT1[i].Im) / (FFT2[i].Re * FFT2[i].Re + FFT2[i].Im * FFT2[i].Im)), 2);
		LSD = sqrt(LSD);
		double SDR = 10 * log10f(mse1[channel] / mse[channel]);
		double SISDR = 10 * log10f(alpha*alpha*mse1[channel] / mse2[channel]);
		double PSNR = 10 * log10f(maxs[channel] * maxs[channel] / mse[channel]);
		printf("------------------Channel %d------------------\n",channel+1);
		printf("MSE = %12.9lf\n", mse[channel]);
		printf("SNR = %12.9lf dB\n", SDR);
		printf("SI-SNR = %12.9lf dB\n", SISDR);
		printf("PSNR = %12.9lf dB\n", PSNR);
		printf("LSD = %12.9lf dB\n", LSD);
		free(FFT1);
		free(FFT2);
	}

	for (int k = 0; k < header1.channels; k++)
		free(samples1[k]);
	// cleanup before quitting
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
