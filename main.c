#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <signal.h>
#include "myProto.h"

/*
	1. Process 0 reads N, K, D, tCount
	2. Process 0 reads the text file and initializes the array of N object(structs).
	3. Procees 0 calculates chunk_Size (num of t values to iterate over).
	4. Process 0 sends chunk_Size, N, K, D, tCount to other processes (MPI_Pack).
	5. Process 0 sends array to processes.
	6. start = rank *chunk_size.
	7. end = start + chunk_size.
	8. For (i = start ... end) { 	
			7.1. Calculate points using CUDA (Store in a local matrix).
			7.2 initialize local indices array (using OpenMP).
			7.3. Check for proximity criteria using CUDA (store indices in a local array).
			7.4. Copy indices array from GPU to CPU. 
	}
	9. Gather all indices to an array of arrays at process 0.
	10. Write values to output.txt file.
*/


/* create MPI_Point_Type */
void create_mpi_point_type(MPI_Datatype *MPI_Point_Type) {
    Point point;

    int point_block_lengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint point_displacements[5];
    MPI_Datatype point_types[5] = {MPI_INT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};

    /* Calculate Point displacements */
    point_displacements[0] = (char *)&point.id - (char *)&point;
    point_displacements[1] = (char *)&point.x1 - (char *)&point;
    point_displacements[2] = (char *)&point.x2 - (char *)&point;
    point_displacements[3] = (char *)&point.a - (char *)&point;
    point_displacements[4] = (char *)&point.b - (char *)&point;
    
    /* Create the MPI struct point type */
    MPI_Type_create_struct(5, point_block_lengths, point_displacements, point_types, MPI_Point_Type);
    MPI_Type_commit(MPI_Point_Type);
}

int main(int argc, char *argv[])
{
	/* variables decleration */
	float D, t1, t2;
	int rank, size, i, N, K, tCount;
	Point *points_array;
    MPI_Status status;

	/* MPI initialization */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* define MPI point type for the defined struct (Point) */
	MPI_Datatype MPI_Point_Type;
	create_mpi_point_type(&MPI_Point_Type);
	
    if (rank == MASTER) {	
    	//t1 = MPI_Wtime();

        /* read first parameters - N, K, D, tCount */
        if (readParametersValues(&N, &K, &D, &tCount) == 1)
        {
            printf("Error readind from file.\n");
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }

        /* allocate memory for struct Point array */
        points_array = (Point *)Malloc(sizeof(Point), N);
        
        /* initialize array of Point objects */
        initObjectsArray(points_array, N);
        
        t1 = MPI_Wtime();
        /* send parameters and objects array to other processes */
        sendParameters(MASTER, size, &N, &K, &D, &tCount);
        
    } else {
    	/* receive parameters from master process */
    	receiveParameters(&N, &K, &D, &tCount);
    	
    	/* allocate memory for struct Point array */
		points_array = (Point *)Malloc(sizeof(Point), N);
    }
    
    /* broadcast array to all processes */
    MPI_Bcast(points_array, N, MPI_Point_Type, MASTER, MPI_COMM_WORLD);
    
    /* calculate local range of t values to evaluate for each process */ 
    int chunkSize, remainder, start, end;
    calcTRange(rank, tCount, size, &chunkSize, &remainder, &start, &end);
    
	/* Allocate memory for local indices array */
	int *local_indices = (int *)Malloc(sizeof(int), NUM_SATISFY * chunkSize);
	
	/* Allocate memory for local t values array */
	float *local_t = (float *)Malloc(sizeof(float), chunkSize);
	
	/* Allocate memory for local satisfied values array */
	int *local_satisfied = (int *)Malloc(sizeof(int), chunkSize);
   	
	/* evaluate local t values */
    evaluateTValues(points_array, local_indices, local_satisfied, local_t, start, end, N, K, D, tCount);
    
	/* Allocate memory for global indices array */
	int *global_indices_array = (int *)Malloc(sizeof(int), NUM_SATISFY * tCount);
	
	/* Allocate memory for global t array */
	float *global_t = (float *)Malloc(sizeof(float), tCount);
	
	/* Allocate memory for global indices array */
	int *global_satisfied = (int *)Malloc(sizeof(int), tCount);

	/* Gather local_indices arrays to MASTER process */
	MPI_Gather(local_indices, chunkSize * NUM_SATISFY, MPI_INT, global_indices_array, chunkSize * NUM_SATISFY, MPI_INT, MASTER, MPI_COMM_WORLD);

	/* Gather local_t arrays to MASTER process */
	MPI_Gather(local_t, chunkSize, MPI_FLOAT, global_t, chunkSize, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

	/* Gather local_satisfied arrays to MASTER process */
	MPI_Gather(local_satisfied, chunkSize, MPI_INT, global_satisfied, chunkSize, MPI_INT, MASTER, MPI_COMM_WORLD);
	
	/* MASTER writes results to output file */
	if (rank == MASTER) {
		writeToFile(global_indices_array, global_t, global_satisfied, tCount);
		t2 = MPI_Wtime();
		
		printf("Time = %f seconds\n", t2 - t1);
	}
    
    /* Free all located memory */
	free(points_array);
	free(local_indices);
	free(local_t);
	free(local_satisfied);
	free(global_indices_array);
	free(global_t);
	free(global_satisfied);
	MPI_Type_free(&MPI_Point_Type);
	
    MPI_Finalize();
    return 0;
}

