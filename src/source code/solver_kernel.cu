#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

__device__ void lock(int *mutex)
{
	while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex)
{
	atomicExch(mutex, 0);
}

__global__ void solver_kernel_naive(float *element, float* global_diff, int * mutex)
{
	__shared__ float tRes[TILE_SIZE * TILE_SIZE];

    int rowNumber = blockDim.y * blockIdx.y + threadIdx.y;
    int columnNumber = threadIdx.x;
    int tId = threadIdx.y * TILE_SIZE + threadIdx.x;
    int index = 0.0f;
    float temp = 0.0f;
    float diff = 0.0f;

    for ( int i = columnNumber; i < GRID_DIMENSION; i += TILE_SIZE )
	{
        if ( i > 0 && i < GRID_DIMENSION - 1 && rowNumber > 0 && rowNumber < GRID_DIMENSION - 1 )
		{
            index = rowNumber * GRID_DIMENSION + i;
            temp = element[index];
            element[index] = 0.20 * (element[index] +
								element[index - GRID_DIMENSION] +
								element[index + GRID_DIMENSION] +
								element[index + 1] +
								element[index - 1]);
            diff += fabsf(element[index] - temp);
        }
    }
    tRes[tId] = diff;
    __syncthreads();
	 
    unsigned int i = TILE_SIZE * TILE_SIZE / 2;
	while ( i != 0 )
	{
		if ( tId < i ) 
			tRes[tId] += tRes[tId + i];
		__syncthreads();
		i /= 2;
	}

    if ( tId == 0 ) 
	{
        lock(mutex);
        *global_diff += tRes[0];
        unlock(mutex);
    }
}

__global__ void solver_kernel_optimized(float * element, float * global_diff, int * mutex)
{
	__shared__ float tRes[TILE_SIZE * TILE_SIZE];
	
    int rowNumber = blockDim.y * blockIdx.y + threadIdx.y;
    int columnNumber = threadIdx.x;
    int tId = threadIdx.y * TILE_SIZE + threadIdx.x;
    int index = 0.0f;
    float temp = 0.0f;
    float diff = 0.0f;

    for ( int i = columnNumber; i < GRID_DIMENSION; i += TILE_SIZE )
	{
        if ( i > 0 && i < GRID_DIMENSION - 1 && rowNumber > 0 && rowNumber < GRID_DIMENSION - 1 )
		{
            index = rowNumber * GRID_DIMENSION + i;
            temp = element[index];
            element[index] = 0.20 * (tex1Dfetch(tElements, index) +
							tex1Dfetch(tElements, index - GRID_DIMENSION) +
							tex1Dfetch(tElements, index + GRID_DIMENSION) +
							tex1Dfetch(tElements, index + 1) +
							tex1Dfetch(tElements, index - 1));
            diff += fabsf(element[index] - temp);
        }
    }
    tRes[tId] = diff;
    __syncthreads();
	 
    unsigned int i = TILE_SIZE * TILE_SIZE / 2;
	while ( i != 0 )
	{
		if ( tId < i ) 
			tRes[tId] += tRes[tId + i];
		__syncthreads();
		i /= 2;
	}

    if ( tId == 0 ) 
	{
        lock(mutex);
        *global_diff += tRes[0];
        unlock(mutex);
    }
}

#endif /* _MATRIXMUL_KERNEL_H_ */
