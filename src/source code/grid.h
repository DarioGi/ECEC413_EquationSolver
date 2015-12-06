#ifndef __GRID_STRUCT__
#define __GRID_STRUCT_

#define GRID_DIMENSION 1024
#define TOLERANCE 0.01 // Tolerance value for convergence
#define UPPER_BOUND_ON_GRID_VALUE 100// The largest value in the grid
#define TILE_SIZE 16

typedef struct grid_struct{
	int num_elements; // Number of points in the grid
	int dimension; // Dimension of the  n x n grid
	float *element;
} GRID_STRUCT;

#endif
