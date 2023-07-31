build:
	mpicxx -fopenmp -g -c main.c -o main.o
	mpicxx -fopenmp -g -c cFunctions.c -o cFunctions.o
	nvcc -I./Common -g -G -gencode arch=compute_61,code=sm_61 -c cudaFunctions.cu -o cudaFunctions.o
	mpicxx -fopenmp -o mpiCudaOpemMP  main.o cFunctions.o cudaFunctions.o  -L/usr/local/cuda/lib -L/usr/local/cuda/lib64 -lcudart
clean:
	rm -f *.o ./mpiCudaOpemMP

run:
	mpiexec -np 1 ./mpiCudaOpemMP

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP
	
debug:
	mpirun -np 2 valgrind --leak-check=full --show-leak-kinds=all -s ./mpiCudaOpemMP
