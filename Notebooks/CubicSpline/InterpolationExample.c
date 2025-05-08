#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <io.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wave.h"
#define TRUE 1 
#define FALSE 0

// WAVE header structure

unsigned char buffer4[4];
unsigned char buffer2[2];

FILE* ptr;
FILE* fptr;
char* filename;
struct HEADER header;


#define BUFFER_SIZE 20000
#define SLOPE_LENGTH 4


typedef struct {
	float* samples;
	int length;
	float sample_rate;
	int num_channels;
} AudioBuffer;

typedef struct {
	float threshold_ratio;
	float gain_linear;
	int buffer_size;
} ClipFixParams;

// Function prototypes
void clipfix_process(AudioBuffer* audio, ClipFixParams* params, float peak_level);
void processBuffer(float* buffer, int bufferLength, float threshold, int slopeLength);
void interpolate(float* buffer, int t0, int t1, int slopeLength);
float db_to_linear(float db);


int main(int argc, char** argv) {
	float** arrf = NULL;
	filename = (char*)malloc(sizeof(char) * 1024);
	if (filename == NULL) {
		printf("Error in malloc\n");
		exit(1);
	}

	// get file path
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {

		strcpy_s(filename, 1024,cwd);

		// get filename from command line
		if (argc < 5) {
			printf("No enough arguments specified\n");
			return -1;
		}

		strcat_s(filename,1024,"\\");
		strcat_s(filename,1024,argv[1]);
		//printf("Input file: %s\n", filename);
	}

	// open file
	fopen_s(&ptr, filename, "rb");
	if (ptr == NULL) {
		printf("Error opening input file\n");
		exit(1);
	}

	int read = 0;
	// the valid amplitude range for values based on the bits per sample
	long long low_limit = 0l;
	long long high_limit = 0l;
	// read header parts

	read = fread(header.riff, sizeof(header.riff), 1, ptr);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);

	// convert little endian to big endian 4 byte int
	header.overall_size = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(header.wave, sizeof(header.wave), 1, ptr);
	read = fread(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, ptr);
	read = fread(buffer4, sizeof(buffer4), 1, ptr);

	// convert little endian to big endian 4 byte integer
	header.length_of_fmt = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer2, sizeof(buffer2), 1, ptr); 
	header.format_type = buffer2[0] | (buffer2[1] << 8);
	read = fread(buffer2, sizeof(buffer2), 1, ptr);
	header.channels = buffer2[0] | (buffer2[1] << 8);
	read = fread(buffer4, sizeof(buffer4), 1, ptr);
	header.sample_rate = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);
	header.byterate = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer2, sizeof(buffer2), 1, ptr);
	header.block_align = buffer2[0] |
		(buffer2[1] << 8);
	read = fread(buffer2, sizeof(buffer2), 1, ptr);
	header.bits_per_sample = buffer2[0] |
		(buffer2[1] << 8);

	read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, ptr);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);

	header.data_size = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	arrf = (float**)malloc(header.channels * sizeof(float*));
	// calculate no.of samples
	long num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);

	long size_of_each_sample = (header.channels * header.bits_per_sample) / 8;

	// calculate duration of file
	float duration_in_seconds = (float)header.overall_size / header.byterate;

	// read each sample from data chunk if PCM
	long bytes_in_each_channel = (size_of_each_sample / header.channels);
	if (header.format_type == 1) { // PCM
		long i = 0;
		char* data_buffer = (char*)malloc(size_of_each_sample);
		int  size_is_correct = TRUE;

		// make sure that the bytes-per-sample is completely divisible by num.of channels
		if ((bytes_in_each_channel * header.channels) != size_of_each_sample) {
			size_is_correct = FALSE;
		}

		if (size_is_correct) {

			switch (header.bits_per_sample) {
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
			for (int k = 0; k < header.channels; k++)
				arrf[k] = (float*)malloc(num_samples * sizeof(float));
			for (i = 1; i <= num_samples; i++) {
				read = fread(data_buffer, size_of_each_sample, 1, ptr);
				if (read == 1) {
					// dump the data read
					unsigned int  xchannels = 0;
					int data_in_channel = 0;
					int offset = 0; // move the offset for every iteration in the loop below
					for (xchannels = 0; xchannels < header.channels; xchannels++) {
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
				//} // 	for (i =1; i <= num_samples; i++) {
			} // 	if (size_is_correct) { 
			free(data_buffer);
		} // if (c == 'Y' || c == 'y') { 
	} //  if (header.format_type == 1) { 
	fclose(ptr);

	// cleanup before quitting
	free(filename);
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
	params.buffer_size = num_samples;
	params.gain_linear = db_to_linear(strtof(argv[4], NULL));
	params.threshold_ratio = strtof(argv[3],NULL);
	for (int chan = 0; chan < header.channels; chan++)
	{
		mybuffer[chan].length = num_samples;
		mybuffer[chan].samples = arrf[chan];
		mybuffer[chan].num_channels = 1;
		mybuffer[chan].sample_rate = header.sample_rate;
		clipfix_process(&mybuffer[chan], &params, mval[chan]);
	}
	strcat_s(cwd, 1024, "\\");
	strcat_s(cwd, 1024, argv[2]);
	//printf("Output file: %s", cwd);
	_unlink(cwd);
	fopen_s(&fptr, cwd, "wb");
	if (fptr == NULL) {
		printf("Error opening output file\n");
		exit(1);
	}
	int written = 0;
	fwrite(&header, (size_t)sizeof(header), 1, fptr);
	int var;
	int value;
	for (int j = 0; j < num_samples; j++)
	{
		for (int m = 0; m < header.channels; m++)
		{
			value = floor(arrf[m][j] * abs(low_limit));
			if (bytes_in_each_channel == 1)
				var = value & 0xff;
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

float db_to_linear(float db) {
	return powf(10.0f, db / 20.0f);
}

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

void process_buffer(float* buffer, int buffer_length, float threshold) {
	int* exit_list = NULL;
	int* return_list = NULL;
	int* exit_list_ptr = NULL;
	int* return_list_ptr = NULL;
	int exit_count = 0;
	int return_count = 0;
	const int last_sample = buffer_length - SLOPE_LENGTH;
	// Detect threshold crossings
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
	// Handle edge case where audio starts clipped
	if (exit_count > 0 && return_count > 0 &&
		fabsf(buffer[SLOPE_LENGTH - 1]) >= threshold) {
		return_list++;
		return_count--;
	}


	// Process found regions
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


void clipfix_process(AudioBuffer* audio, ClipFixParams* params, float peak_level) {
	const float threshold = params->threshold_ratio * peak_level;
	const float gain = params->gain_linear;
	const int total_samples = audio->length;

	// Process in chunks
	for (int i = 0; i < total_samples; i += params->buffer_size) {
		int chunk_size = (i + params->buffer_size > total_samples)
			? total_samples - i
			: params->buffer_size;
		process_buffer(audio->samples + i, chunk_size, threshold,BUFFER_SIZE);
		//process_buffer(audio->samples + i, chunk_size, threshold, total_samples);
	}
	// Apply gain
	for (int i = 0; i < total_samples; i++) {
		audio->samples[i] *= gain;
	}
}