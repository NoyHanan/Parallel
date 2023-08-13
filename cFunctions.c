#include "myProto.h"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>


/* memory allocation */
void* Malloc(size_t type_size, size_t num_elements) {
    void* ptr = malloc(type_size * num_elements);
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}


/* read N, K, D, tCount */
int readParametersValues(int *N, int *K, float *D, int *tCount)
{   
	/* open file */
    FILE *file = fopen("input.txt", "r");
    if (file == NULL)
    {
        printf("Error opening the file.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
	
	if(fscanf(file, "%d %d %f %d", N, K, D, tCount) != 4) return 1;
	
    /* close file */
    fclose(file);
	
	return 0;
}


/* read rows from file, create struct Data and store it in the array */
int initObjectsArray(Point *points_array, int n)
{
	int i, res;
	char buffer[256]; // Buffer to hold the first line
	
	/* open file */
    FILE *file = fopen("input.txt", "r");
    if (file == NULL)
    {
        printf("Error opening the file.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
    
    /* Read and discard the first line */
    if (fgets(buffer, sizeof(buffer), file) == NULL) {
        printf("Error reading the first line of the file.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
	
	for (i = 0; i < n; i++) {
		Point point;
		
		res = fscanf(file, "%d %f %f %f %f", &point.id, &point.x1, &point.x2, &point.a, &point.b);
		if (res != 5) {
			return 1;
		} else {
			points_array[i] = point;
		}
	}
	
    /* close file */
    fclose(file);
	
	return 0;
}


/* send parameters from master process to other processes */
void sendParameters(int process, int numProc, int *N, int *K, float *D, int *tCount)
{
	int i, position = 0;
	int bufSize = 3 * sizeof(int) + 1 * sizeof(float);
	char *buf = (char *)malloc(bufSize);
	
	/* pack variables */
	MPI_Pack(N, 1, MPI_INT, buf, bufSize, &position, MPI_COMM_WORLD);
    MPI_Pack(K, 1, MPI_INT, buf, bufSize, &position, MPI_COMM_WORLD);
    MPI_Pack(D, 1, MPI_FLOAT, buf, bufSize, &position, MPI_COMM_WORLD);
    MPI_Pack(tCount, 1, MPI_INT, buf, bufSize, &position, MPI_COMM_WORLD);
    
    #pragma omp parallel for
    for (i = 0; i < numProc; i++) {
    	if (i != process)
    		MPI_Send(buf, position, MPI_PACKED, i, 0, MPI_COMM_WORLD);
    }
    
    free(buf);
}


/* receive parameters from master process */
void receiveParameters(int *N, int *K, float *D, int *tCount)
{
    int position = 0;
	int bufSize = 3 * sizeof(int) + 1 * sizeof(float);
	char *buf = (char *)malloc(bufSize);
	MPI_Status status;
	
	/* Recieve the packed message */
	MPI_Recv(buf, bufSize, MPI_PACKED, 0, 0, MPI_COMM_WORLD, &status);
	
	/* unpack */
    MPI_Unpack(buf, bufSize, &position, N, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bufSize, &position, K, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bufSize, &position, D, 1, MPI_FLOAT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bufSize, &position, tCount, 1, MPI_INT, MPI_COMM_WORLD);
    
    free(buf);
}


/* send results to MASTER */
void sendIndices(int *local_indices, float *t)
{
	int i, position = 0;
	int bufSize = NUM_SATISFY * sizeof(int) + 1 * sizeof(float);
	char *buf = (char *)malloc(bufSize);
	
	/* pack variables */
	MPI_Pack(local_indices, NUM_SATISFY, MPI_INT, buf, bufSize, &position, MPI_COMM_WORLD);
    MPI_Pack(t, 1, MPI_FLOAT, buf, bufSize, &position, MPI_COMM_WORLD);
    
   	MPI_Send(buf, position, MPI_PACKED, MASTER, 0, MPI_COMM_WORLD);
    		
    free(buf);
}


/* receive results from slave processes */
void receiveIndices(int *local_indices, float *t)
{
	int position = 0;
	int bufSize = NUM_SATISFY * sizeof(int) + 1 * sizeof(float);
	char *buf = (char *)malloc(bufSize);
	MPI_Status status;
	
	/* Recieve the packed message */
	MPI_Recv(buf, bufSize, MPI_PACKED, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	
	/* unpack */
    MPI_Unpack(buf, bufSize, &position, local_indices, NUM_SATISFY, MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buf, bufSize, &position, t, 1, MPI_FLOAT, MPI_COMM_WORLD);
    
    free(buf);
}


/* calculate t range values */
void calcTRange(int rank, int tCount, int size, int *chunkSize, int *remainder, int *start, int *end)
{
	*remainder = tCount % size;
    *chunkSize = tCount / size + ((rank < *remainder) ? *remainder : 0);
    *start = rank * *chunkSize + ((rank < *remainder) ? rank : *remainder);
    *end = *start + *chunkSize;
}


/* evaluate local t values */
void evaluateTValues(FILE *file, Point *points_array, int rank, int start, int end, int N, int K, float D, int tCount) {
	int i, index = 0, satisfy = -1;
	
	#pragma omp parallel for
    for (i = start; i < end; i++) {
    	/* calculate t */
    	float t = (float)(2 * i) / tCount - 1;
    	
    	/* Allocate memory for local indices array */
		int *local_indices = (int *)Malloc(sizeof(int), NUM_SATISFY);
		
		/* check proximity criteria */
    	int res = computeOnGPU(local_indices, points_array, N, K, D, t);
		
		if (res == 0) {
			if (rank == MASTER)
				/* write local results to file */
				#pragma omp critical
				writeToFile(file, local_indices, t);
			else 
				/* send results to MASTER */
				sendIndices(local_indices, &t);
			
			satisfy = 0;
    	}
    		
    	free(local_indices);
    }
    
    if (satisfy == -1)
    	fprintf(file, "There were no %d points found for any t.", NUM_SATISFY);
}


/* master receives and writes to output file all slaves results */
void handleSlaveResults(FILE *file)
{
	int flag;
	MPI_Status status;
	
	do {
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
		if (flag) {
			float t;
			int *local_indices = (int *)Malloc(sizeof(int), NUM_SATISFY);
			
			/* receive results from slave processes */
			receiveIndices(local_indices, &t);
			
			/* write to file */
			writeToFile(file, local_indices, t);
			
			free(local_indices);
		}
	} while(flag);
}


/* write local results to output file */
void writeToFile(FILE *file, int *local_indices, float t)
{   
    int j;
    
    fprintf(file, "Points ");
    for (j = 0; j < NUM_SATISFY - 1; j++) {
        fprintf(file, "pointID%d, ", local_indices[j]);
    }
    fprintf(file, "pointID%d ", local_indices[j]);
    fprintf(file, "satisfy Proximity Criteria at t = %f\n", t);
}


