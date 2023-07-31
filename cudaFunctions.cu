#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <omp.h>
#include <stdarg.h>
#include "myProto.h"

#define NUM_THREADS 256

/* calculate x */
__device__ float calculate_x_device(Point p, double t) {
    return ((p.x2 - p.x1) / 2) * sin(t * M_PI / 2) + (p.x2 + p.x1) / 2;
}


/* calculate x */
__device__ float calculate_y_device(Point p, double x) {
    return p.a * x + p.b;
}


/* CPU function to calculate distance between every coordinate */
__device__ float calculateDistanceDevice(Point p1, Point p2, float t) {
    float x1 = calculate_x_device(p1, t);
    float y1 = calculate_y_device(p1, x1);

    float x2 = calculate_x_device(p2, t);
    float y2 = calculate_y_device(p2, x2);

    return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}


/* kernel function to check proximity criteria for each point */
__global__ void checkProximityCriteria(Point *points, int *results, int *globalCounter, int numPoints, int K, float D, float t) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

	for (int i = index; i < numPoints && *globalCounter < NUM_SATISFY; i += stride) {
	    int count = 0;
	    Point p1 = points[i];
	    
		for (int j = 0; j < numPoints && *globalCounter < NUM_SATISFY && count < K; j++) {
			Point p2 = points[j];
        	float distance = calculateDistanceDevice(p1, p2, t);
        	if (i != j && distance < D) {
	            count++;
	        }
	    }
	    
	    if (count >= K) {
	    	results[i] = 1;
	    	atomicAdd(globalCounter, 1);
	    } else {
	    	results[i] = 0;
	    }
	}
}


/* find 3 points that satisfy the criteria and insert them to local indices array */
void fillIndicesArray(int *results, int *indices_array, int numPoints)
{
	int i, count = 0;
	
	for (i = 0; i < numPoints; i++) {
		if (results[i] == 1) {
			indices_array[count] = i;
			count++;
			
			if (count >= NUM_SATISFY)
				break;
		}
	}
}


/* GPU "controller" function to evaluate Proximity Criteria for each point */
int computeOnGPU(int *local_indices, Point *h_points, int N, int K, float D, float t) {
    Point *d_points;
    int *d_results;
    int *d_global_counter;
    int *h_results;
    cudaError_t err;
    
    /* Calculate required blocks */
    int numBlocks = (N + NUM_THREADS - 1) / NUM_THREADS;

	/* Allocate memory for points on the device */
    err = cudaMalloc((void**)&d_points, N * sizeof(Point));
    if (err != cudaSuccess) {
        printf("GPU - Failed to allocate device memory for points: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    /* Copy points from host to device */
    err = cudaMemcpy(d_points, h_points, N * sizeof(Point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("GPU - Failed to copy points from host to device: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    /* Allocate memory for results on the device */
    err = cudaMalloc((void**)&d_results, N * sizeof(int));
    if (err != cudaSuccess) {
        printf("GPU - Failed to allocate device memory for results: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    /* Allocate memory for the global counter on the device */
	err = cudaMalloc((void**)&d_global_counter, sizeof(int));
	if (err != cudaSuccess) {
		printf("GPU - Failed to allocate device memory for counter: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE); 
	}

	/* Set the counter in the device memory to 0 */
	err = cudaMemset(d_global_counter, 0, sizeof(int));;
	if (err != cudaSuccess) {
		printf("GPU - Failed to initialize device memory for counter: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    /* Run the checkProximityCriteria kernel */
    checkProximityCriteria<<<numBlocks, NUM_THREADS>>>(d_points, d_results, d_global_counter, N, K, D, t);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error running checkProximityCriteria: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);;
    }

    /* Copy counter from device to host */
    int counter;
    err = cudaMemcpy(&counter, d_global_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("GPU - Failed to copy counter from device to host: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
    }
    
    /* free d_global_counter */
    err = cudaFree(d_global_counter);
    if (err != cudaSuccess) {
        printf("Error freeing d_global_counter: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	/* Allocate memory for results on host */
	h_results = (int *)malloc(N * sizeof(int));
	if (h_results == NULL) {
		printf("GPU - Failed to copy results from device to host: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    /* Copy results from device to host */
    err = cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("GPU - Failed to copy results from device to host: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    /* free d_results */
    err = cudaFree(d_results);
    if (err != cudaSuccess) {
        printf("Error freeing d_results: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    /* find NUM_SATISFY points that satisfy the criteria and insert them to local indices array */
    if (counter == NUM_SATISFY)
    	fillIndicesArray(h_results, local_indices, N);

    /* Free device memory */
	free(h_results);
    
    return counter < NUM_SATISFY ? 1 : 0;
}

