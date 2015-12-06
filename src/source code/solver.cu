/* 
Code for the equation solver. 
Author: Naga Kandasamy 
Date: 11/25/2015 
*/

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h" // This file defines the grid data structure
#include <sys/time.h>
#include <limits.h>

texture<float> tElements;
#include "solver_kernel.cu"

extern "C" void compute_gold(GRID_STRUCT *);


/* This function prints the grid on the screen */
void 
display_grid(GRID_STRUCT *my_grid)
{
	for(int i = 0; i < my_grid->dimension; i++)
		for(int j = 0; j < my_grid->dimension; j++)
			printf("%f \t", my_grid->element[i * my_grid->dimension + j]);
   		
		printf("\n");
}


/* This function prints out statistics for the converged values, including min, max, and average. */
void 
print_statistics(GRID_STRUCT *my_grid)
{
		// Print statistics for the CPU grid
		float min = INFINITY;
		float max = 0.0;
		double sum = 0.0; 
		for(int i = 0; i < my_grid->dimension; i++){
			for(int j = 0; j < my_grid->dimension; j++){
				sum += my_grid->element[i * my_grid->dimension + j]; // Compute the sum
				if(my_grid->element[i * my_grid->dimension + j] > max) max = my_grid->element[i * my_grid->dimension + j]; // Determine max
				if(my_grid->element[i * my_grid->dimension + j] < min) min = my_grid->element[i * my_grid->dimension + j]; // Determine min
				 
			}
		}

	printf("AVG: %f \n", sum/(float)my_grid->num_elements);
	printf("MIN: %f \n", min);
	printf("MAX: %f \n", max);

	printf("\n");
}


/* Calculate the differences between grid elements for the various implementations. */
void compute_grid_differences(GRID_STRUCT *grid_1, GRID_STRUCT *grid_2)
{
    float diff;
    int dimension = grid_1->dimension;
    int num_elements = dimension*dimension;

    diff = 0.0;
    for(int i = 0; i < grid_1->dimension; i++){
        for(int j = 0; j < grid_1->dimension; j++){
            diff += fabsf(grid_1->element[i * dimension + j] - grid_2->element[i * dimension + j]);
        }
    }
    printf("Average difference in grid elements for Gauss Seidel and Jacobi methods = %f. \n", \
            diff/num_elements);

}



/* This function creates a grid of random floating point values bounded by UPPER_BOUND_ON_GRID_VALUE */
void 
create_grids(GRID_STRUCT *grid_for_cpu, GRID_STRUCT *grid_for_gpu, GRID_STRUCT *grid_for_gpu_text)
{
	printf("Creating a grid of dimension %d x %d. \n", grid_for_cpu->dimension, grid_for_cpu->dimension);
	grid_for_cpu->element = (float *)malloc(sizeof(float) * grid_for_cpu->num_elements);
	grid_for_gpu->element = (float *)malloc(sizeof(float) * grid_for_gpu->num_elements);
	grid_for_gpu_text->element = (float *)malloc(sizeof(float) * grid_for_gpu_text->num_elements);

	srand((unsigned)time(NULL)); // Seed the the random number generator 
	
	float val;
	for(int i = 0; i < grid_for_cpu->dimension; i++)
		for(int j = 0; j < grid_for_cpu->dimension; j++){
			val =  ((float)rand()/(float)RAND_MAX) * UPPER_BOUND_ON_GRID_VALUE; // Obtain a random value
			grid_for_cpu->element[i * grid_for_cpu->dimension + j] = val; 	
			grid_for_gpu->element[i * grid_for_gpu->dimension + j] = val; 
			grid_for_gpu_text->element[i * grid_for_gpu_text->dimension + j] = val; 			
		}
}


/* Edit this function skeleton to solve the equation on the device. Store the results back in the my_grid->element data structure for comparison with the CPU result. */
void compute_on_device(GRID_STRUCT *my_grid)
{
	float *dElements = NULL;
	float *dDiff = NULL;
	
	cudaMalloc((void**)&dElements, my_grid->num_elements * sizeof(float));
	cudaMemcpy(dElements, my_grid->element, my_grid->num_elements * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dDiff, sizeof(float));
	cudaMemset(dDiff, 0, sizeof(float));
	
	int *mutex = NULL;
    cudaMalloc((void **)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));

	dim3 thread_block(TILE_SIZE, TILE_SIZE, 1);
	dim3 grid(1,(my_grid->dimension) / TILE_SIZE);

	float temp = INT_MAX;
    struct timeval start, stop;
    gettimeofday(&start, NULL);
	unsigned int count = 0;
	
	while ( temp / (my_grid->dimension * my_grid->dimension) >= TOLERANCE )
	{
		cudaMemset(dDiff, 0, sizeof(float));
		solver_kernel_naive<<<grid, thread_block>>>(dElements, dDiff, mutex);
		cudaThreadSynchronize();
		cudaError_t err = cudaGetLastError();
		if ( cudaSuccess != err ) 
		{
			fprintf(stderr, "Kernel execution failed: %s.\n", cudaGetErrorString(err));
			return;
		}
		cudaMemcpy(&temp, dDiff, sizeof(float), cudaMemcpyDeviceToHost);
		count++;
	}

	gettimeofday(&stop, NULL);
    printf("Iterations: %d, GPU Global Memory runtime: %0.5fs.\n", count, (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	cudaMemcpy(my_grid->element, dElements, my_grid->num_elements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dElements);
	cudaFree(dDiff);
}

// Using texture
void compute_on_device_text(GRID_STRUCT *my_grid)
{
	float *dElements = NULL;
	float *dDiff = NULL;
	
	cudaMalloc((void**)&dElements, my_grid->num_elements * sizeof(float));
	cudaMemcpy(dElements, my_grid->element, my_grid->num_elements * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dDiff, sizeof(float));
	cudaMemset(dDiff, 0, sizeof(float));
	
	cudaBindTexture(NULL, tElements, dElements, my_grid->num_elements * sizeof(float));
	int *mutex = NULL;
    cudaMalloc((void **)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));

	dim3 thread_block(TILE_SIZE, TILE_SIZE, 1);
	dim3 grid(1,(my_grid->dimension) / TILE_SIZE);

	float temp = INT_MAX;
    struct timeval start, stop;
    gettimeofday(&start, NULL);
	unsigned int count = 0;
	
	while ( temp / (my_grid->dimension * my_grid->dimension) >= TOLERANCE )
	{
		cudaMemset(dDiff, 0, sizeof(float));
		solver_kernel_optimized<<<grid, thread_block>>>(dElements, dDiff, mutex);
		cudaThreadSynchronize();
		cudaError_t err = cudaGetLastError();
		if ( cudaSuccess != err ) 
		{
			fprintf(stderr, "Kernel execution failed: %s.\n", cudaGetErrorString(err));
			return;
		}
		cudaMemcpy(&temp, dDiff, sizeof(float), cudaMemcpyDeviceToHost);
		count++;
	}

	gettimeofday(&stop, NULL);
    printf("Iterations: %d, GPU Textured Memory runtime: %0.5fs.\n", count, (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	cudaMemcpy(my_grid->element, dElements, my_grid->num_elements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(dElements);
	cudaFree(dDiff);
}

/* The main function */
int main(int argc, char **argv)
{	
	/* Generate the grid */
	GRID_STRUCT *grid_for_cpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure
	GRID_STRUCT *grid_for_gpu = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure
	GRID_STRUCT *grid_for_gpu_text = (GRID_STRUCT *)malloc(sizeof(GRID_STRUCT)); // The grid data structure
	
	grid_for_cpu->dimension = GRID_DIMENSION;
	grid_for_cpu->num_elements = grid_for_cpu->dimension * grid_for_cpu->dimension;
	grid_for_gpu->dimension = GRID_DIMENSION;
	grid_for_gpu->num_elements = grid_for_gpu->dimension * grid_for_gpu->dimension;
	grid_for_gpu_text->dimension = GRID_DIMENSION;
	grid_for_gpu_text->num_elements = grid_for_gpu_text->dimension * grid_for_gpu_text->dimension;

 	create_grids(grid_for_cpu, grid_for_gpu, grid_for_gpu_text);
	
	printf("Using the cpu to solve the grid. \n");
	struct timeval start, stop;
    gettimeofday(&start, NULL);
	compute_gold(grid_for_cpu);  // Use CPU to solve 
	gettimeofday(&stop, NULL);
    printf("CPU runtime: %0.5f s. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

	// Use the GPU to solve the equation
	compute_on_device(grid_for_gpu);
	
	// Use the GPU and textured memory to solve the equation.
	compute_on_device_text(grid_for_gpu_text);
	
	// Print key statistics for the converged values
	printf("CPU: \n");
	print_statistics(grid_for_cpu);

	printf("GPU: \n");
	print_statistics(grid_for_gpu);
	
	printf("GPU textured: \n");
	print_statistics(grid_for_gpu_text);
	
    /* Compute grid differences. */
    compute_grid_differences(grid_for_cpu, grid_for_gpu);
	
	printf("Textured version: ");
	compute_grid_differences(grid_for_cpu, grid_for_gpu_text);

	free((void *)grid_for_cpu->element);	
	free((void *)grid_for_cpu); // Free the grid data structure 
	
	free((void *)grid_for_gpu->element);	
	free((void *)grid_for_gpu); // Free the grid data structure 

	free((void *)grid_for_gpu_text->element);	
	free((void *)grid_for_gpu_text); // Free the grid data structure 
	
	exit(0);
}
