#pragma once
#include <stdio.h>

#define MASTER 0
#define NUM_SATISFY 3
#define PAR_OUTPUT "output.txt"
#define SEQ_OUTPUT "output_seq.txt"

/*****************************
 ***** type definitions ******
 ****************************/
typedef struct {
    int id;
    float x1;
    float x2;
    float a;
    float b;
} Point;


/**************************************
 ***** CPU functions definitions ******
 **************************************/
/* memory allocation */
void *Malloc(size_t type_size, size_t num_elements);
 
/* read N, K, D, tCount */
int readParametersValues(int *N, int *K, float *D, int *tCount);

/* read rows from file, create struct Data and store it in the array */
int initObjectsArray(Point *points_array, int n);

/* send parameters from master process to other processes */
void sendParameters(int process, int numProc, int *N, int *K, float *D, int *tCount);

/* receive parameters from master process */
void receiveParameters(int *N, int *K, float *D, int *tCount);

/* calculate t range values */
void calcTRange(int rank, int tCount, int size, int *chunkSize, int *remainder, int *start, int *end);

/* evaluate local t values */
void evaluateTValues(Point *points_array, int *local_indices, int *local_satisfied, float *local_t, int start, int end, int N, int K, float D, int tCount);

/* write results to output file */
void writeToFile(int *global_indices_array, float *global_t, int *global_satisfied, int tCount);

/* CPU function to calculate distance between every coordinate */
float calculateDistanceSeq(Point p1, Point p2, float t);

/* CPU function to check proximity criteria for each point */
void checkProximitySeq(Point *points, int N, int K, float D, int tCount);

/* comapre sequential and parallel results */
int test_file_results(const char *file1, const char *file2);


/**************************************
 ***** GPU functions definitions ******
 **************************************/
/* GPU "controller" function to evaluate Proximity Criteria for each point */
int computeOnGPU(int *local_indices, Point *h_points, int N, int K, float D, float t);

/* find NUM_SATISFY points that satisfy the criteria and insert them to local indices array */
void fillIndicesArray(int *results, int *indices_array, int size);

















