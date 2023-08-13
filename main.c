#include <mpi.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <signal.h>
#include "myProto.h"

/*
	1. Process Master reads N, K, D, tCount
	2. Process Master sends N, K, D, tCount to other processes
	3. Process Master reads the text file and initializes the array of N Points.
	5. Process 0 sends array to processes.
	6. start = rank *chunk_size.
	7. end = start + chunk_size.
	8. For (i = start ... end) { 	
			8.1. Calculate coordinates, distances and Check for proximity criteria using CUDA.
		}
	9. MASTER process writes matching results to output file, slave processes send results to MASTER.
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
    	t1 = MPI_Wtime();

        /* read first parameters - N, K, D, tCount */
        if (readParametersValues(&N, &K, &D, &tCount) == 1)
        {
            printf("Error readind from file.\n");
            MPI_Abort(MPI_COMM_WORLD, __LINE__);
        }
        
        /* send parameters and objects array to other processes */
        sendParameters(MASTER, size, &N, &K, &D, &tCount);

        /* allocate memory for struct Point array */
        points_array = (Point *)Malloc(sizeof(Point), N);
        
        /* initialize array of Point objects */
        initObjectsArray(points_array, N);
        
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
    
    /* open output file */
    FILE *file = fopen(PAR_OUTPUT, "w");
    if (file == NULL)
    {
        printf("Error opening the output file.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }
   	
	/* evaluate local t values */
    evaluateTValues(file, points_array, rank, start, end, N, K, D, tCount);
    
    MPI_Barrier(MPI_COMM_WORLD);
	
	/* MASTER writes slave processes results to output file */
	if (rank == MASTER) {
		/* master receives and writes to output file all slaves results */
		handleSlaveResults(file);
		
		t2 = MPI_Wtime();
		printf("Time = %f seconds\n", t2 - t1);
	}
    
    /* Free all located memory */
	free(points_array);
	MPI_Type_free(&MPI_Point_Type);
	
    MPI_Finalize();
    return 0;
}

