#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wb.h"

// BLUR_SIZE is 1 because we span three rows, ( Row-1, Row, Row+1 )
#define BLUR_SIZE 1
#define BLOCK_SIZE 16

int xDimension;
int yDimension;

void writeImage(uchar4 *image, char *filename, char *memorytype);

void readImage(char *filename, uchar4 *image);


__global__ void unsharedBlurring(uchar4 *image, uchar4 *imageOutput, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over rows
    for (int x = 0; x < width; x++) {

        uchar4 pixel = make_uchar4(0, 0, 0, 0);
        float4 average = make_float4(0, 0, 0, 0);

        for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
            for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
                int blurRow = x + i;
                int blurCol = col + j;

                // Below, we check the boundary conditions
                if ((blurRow > -1) && (blurRow < width) && (blurCol > -1) && (blurCol < height)) {
                    pixel.x = image[blurRow + blurCol * height].x;
                    pixel.y = image[blurRow + blurCol * height].y;
                    pixel.z = image[blurRow + blurCol * height].z;
                } 
				else {
                    pixel = make_uchar4(0, 0, 0, 0);
                }
                average.x += pixel.x;
		        average.y += pixel.y;
		        average.z += pixel.z;
            }
        }
        // Divide summation to number of pixels
        average.x /= (float) (((BLUR_SIZE*2)+1)*((BLUR_SIZE*2)+1));
        average.y /= (float) (((BLUR_SIZE*2)+1)*((BLUR_SIZE*2)+1));
        average.z /= (float) (((BLUR_SIZE*2)+1)*((BLUR_SIZE*2)+1));

        imageOutput[x + col * height].x = (unsigned char) average.x;
        imageOutput[x + col * height].y = (unsigned char) average.y;
        imageOutput[x + col * height].z = (unsigned char) average.z;
        imageOutput[x + col * height].w = 255;
    }
}

__global__ void sharedBlurring(uchar4 *image, uchar4 *imageOutput, int width, int height) {
    int col = threadIdx.x + blockIdx.x * (blockDim.x - 2 * BLUR_SIZE);
    int row = threadIdx.y + blockIdx.y * (blockDim.y - 2 * BLUR_SIZE);
    uchar4 pixel = make_uchar4(0, 0, 0, 0);
    float4 average = make_float4(0, 0, 0, 0);

	if((row < height + BLUR_SIZE) && (col < width + BLUR_SIZE)) {
		// Allocate shared memory
		__shared__ uchar4 chunk[BLOCK_SIZE + (2 * BLUR_SIZE)][BLOCK_SIZE + (2 * BLUR_SIZE)];

		// Load elements into memory
		int relativeRow = row - BLUR_SIZE;
        int relativeCol = col - BLUR_SIZE;
        if ((relativeRow < height) && (relativeCol < width) && (relativeRow >= 0) && (relativeCol >= 0)) {
            chunk[threadIdx.y][threadIdx.x] = image[relativeRow*width + relativeCol];
        }
        else {
            chunk[threadIdx.y][threadIdx.x] = make_uchar4(0, 0, 0, 0);
        }

		__syncthreads();

		// Filter out-of-bounds threads
		if ((threadIdx.x >= BLUR_SIZE) && (threadIdx.y >= BLUR_SIZE) && (threadIdx.y < blockDim.y - BLUR_SIZE) && (threadIdx.x < blockDim.x - BLUR_SIZE)) {
			
			for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
				for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
					int blurRow = threadIdx.y + i;
					int blurCol = threadIdx.x + j;

					// Below, we check the boundary conditions
					if ((blurRow >= -1) && (blurRow < height) && (blurCol >= -1) && (blurCol < width)) {
                        pixel.x = chunk[blurRow][blurCol].x;
                        pixel.y = chunk[blurRow][blurCol].y;
                        pixel.z = chunk[blurRow][blurCol].z;
					}
                    else {
                        pixel = make_uchar4(0, 0, 0, 0);
                    }

                    average.x += pixel.x;
			        average.y += pixel.y;
			        average.z += pixel.z;
				}
			}
			// Divide summation to number of pixels
			average.x /= (float) (((BLUR_SIZE*2)+1)*((BLUR_SIZE*2)+1));
			average.y /= (float) (((BLUR_SIZE*2)+1)*((BLUR_SIZE*2)+1));
			average.z /= (float) (((BLUR_SIZE*2)+1)*((BLUR_SIZE*2)+1));

			imageOutput[relativeRow*width + relativeCol].x = (unsigned char) average.x;
			imageOutput[relativeRow*width + relativeCol].y = (unsigned char) average.y;
			imageOutput[relativeRow*width + relativeCol].z = (unsigned char) average.z;
			imageOutput[relativeRow*width + relativeCol].w = 255;
		}
	}
}

/**
 * Host main routine
 */
int main(int argc, char **argv) {
	if(argc != 3) {
		printf("Usage error. Program expects two arguments. \n");
    	printf("Usage: ./imageBlur IMAGENAME BLURTYPE(0 for unshared memory, 1 for shared memory) \n");
		printf("Usage Example: ./imageBlur 1.ppm 0 \n");
    	exit(1);
	}

    // System specifications
    printf("-->\n");
    printf("System Specifications:\n");
    printf("\tAzure NC6\n");
    printf("\tCores: 6\n");
    printf("\tGPU: Tesla K80\n");
    printf("\tMemory: 56 GB\n");
    printf("\tDisk: 380 GB SSD\n");
    printf("-->\n");
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    char *inputImageFile;
    wbImage_t inputImage;

    inputImageFile = argv[1];
    printf("Loading %s...\n", inputImageFile);
    inputImage = wbImport(inputImageFile);

    xDimension = wbImage_getWidth(inputImage);
    yDimension = wbImage_getHeight(inputImage);

    unsigned int imageSize, i;
    uchar4 *deviceImage, *deviceImageOutput, *deviceImageTemp;
    uchar4 *hostImage;
    cudaEvent_t start, stop;
    float3 ms;

    imageSize = xDimension * yDimension * sizeof(uchar4);

    // Create event timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate and load host image
    hostImage = (uchar4 *) malloc(imageSize);
    // Verify that allocations succeeded
    if (hostImage == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    readImage(argv[1], hostImage);

    // Allocate device images
    err = cudaMalloc((void **) &deviceImage, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **) &deviceImageOutput, imageSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device image output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(start, 0);
	if (std::string(argv[2]) == "0") {
        printf("Executing blurring with unshared memory...\n");

		// Copy image to device memory
        printf("Copying image data from the host memory to the CUDA device...\n");
    	err = cudaMemcpy(deviceImage, hostImage, imageSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy image from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        dim3 blocksPerGrid(xDimension / BLOCK_SIZE, 1);
    	dim3 threadsPerBlock(BLOCK_SIZE, 1);
        printf("CUDA kernel launching with {%d, %d} blocks of {%d, %d} threads...\n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

		for (i = 0; i < 100; i++) {
			unsharedBlurring << <blocksPerGrid, threadsPerBlock >> >(deviceImage, deviceImageOutput, xDimension, yDimension);
			err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to launch unsharedBlurring kernel (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
			deviceImageTemp = deviceImage;
			deviceImage = deviceImageOutput;
			deviceImageOutput = deviceImageTemp;
		}
	}
	else {
        printf("Executing blurring with shared memory...\n");

		// Copy image to device memory
        printf("Copying image data from the host memory to the CUDA device...\n");
		err = cudaMemcpy(deviceImage, hostImage, imageSize, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy image from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        dim3 blocksPerGrid(ceil(xDimension/(float)BLOCK_SIZE), ceil(yDimension/(float)BLOCK_SIZE), 1);
    	dim3 threadsPerBlock(BLOCK_SIZE + 2 * BLUR_SIZE, BLOCK_SIZE + 2 * BLUR_SIZE, 1);
        printf("CUDA kernel launching with {%d, %d, %d} blocks of {%d, %d, %d} threads...\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z, threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
        
        for (i = 0; i < 100; i++) {
            sharedBlurring << < blocksPerGrid, threadsPerBlock >> > (deviceImage, deviceImageOutput, xDimension, yDimension);
		    err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to launch sharedBlurring kernel (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
		    deviceImageTemp = deviceImage;
		    deviceImage = deviceImageOutput;
		    deviceImageOutput = deviceImageTemp;
        }
	}

    // Copy results back to host
    err = cudaMemcpy(hostImage, deviceImage, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy images back to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms.x, start, stop);

    //output timings
    printf("Execution time:\n");
    if (std::string(argv[2]) == "0") {
        printf("\tUnshared version: %f\n", ms.x);
    }
    else {
        printf("\tShared version: %f\n", ms.x);
    }

    // Write image
    writeImage(hostImage, argv[1], argv[2]);

    // Free device memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    err = cudaFree(deviceImage);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(deviceImageOutput);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device image output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // Free host memory
    free(hostImage);
    printf("DONE\n");

    return 0;
}

void writeImage(uchar4 *image, char *filename, char *memorytype) {
    FILE *f;

    char* str1;
    if (std::string(memorytype) == "0") {
        str1 = "unshared_output_";
    }
    else {
        str1 = "shared_output_";
    }
    char * str3 = (char *) malloc(1 + strlen(str1)+ strlen(filename) );
    strcpy(str3, str1);
    strcat(str3, filename);
    f = fopen(str3, "wb");
    if (f == NULL) {
        fprintf(stderr, "Error opening 'output.ppm' output file\n");
        exit(1);
    }
    fprintf(f, "P6\n");
    fprintf(f, "%d %d\n%d\n", xDimension, yDimension, 255);
    for (int x = 0; x < xDimension; x++) {
        for (int y = 0; y < yDimension; y++) {
            int i = x + y * yDimension;
            fwrite(&image[i], sizeof(unsigned char), 3, f);
        }
    }
    free(str3);
    fclose(f);
}

void readImage(char *filename, uchar4 *image) {
    FILE *f;
    char temp[256];
    unsigned int w, h, s;

    f = fopen(filename, "rb");
    if (f == NULL) {
        fprintf(stderr, "Error opening input file\n");
        exit(1);
    }
    printf("------\n");
    printf("Image Info:\n");
    fscanf(f, "%s\n", &temp);
    fscanf(f, "%*[^\n]\n");
    printf("\tType: %s\n", &temp);
    fscanf(f, "%d %d\n", &w, &h);
    printf("\tImage Size:%dx%d\n", w, h);
    fscanf(f, "%d\n", &s);
    printf("------\n");

    for (int x = 0; x < xDimension; x++) {
        for (int y = 0; y < yDimension; y++) {
            int i = x + y * yDimension;
            fread(&image[i], sizeof(unsigned char), 3, f);
        }
    }

    fclose(f);
}
