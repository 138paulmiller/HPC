#include <stdio.h>
#include <math.h>
#define part_1 1
#define part_2_3 1
#define part_4 1

__global__
void init(int arr_n, int* arr){
	//compute absolute index of thread from 
	//relative 	grid and block thread index
	int	index = blockIdx.x * blockDim.x + threadIdx.x;
	//number of threads to skip to get to next blocks thread
	//printf("\nBLOCK: %d, THREAD: %d ", blockIdx.x, threadIdx.x);
	int stride = gridDim.x*blockDim.x;
	int i;
	for(i = index; i < arr_n; i+= stride)
		arr[i] = 0;		
}


__global__
void add_i(int arr_n, int* arr){
	//compute absolute index of thread from 
	//relative 	grid and block thread index
	int	index = blockIdx.x * blockDim.x + threadIdx.x;
	//number of threads to skip to get to next blocks thread
	//printf("\nBLOCK: %d, THREAD: %d ", blockIdx.x, threadIdx.x);
	int stride = gridDim.x*blockDim.x;
	int i;
	for(i = index; i < arr_n; i+= stride)
		arr[i] += i;		
}

void  make_arr(int ** arr, int n){
	//c - malloc abstraction to alloc arr from CPU and GPU
	cudaMallocManaged(arr, n*sizeof(int));
}
__global__
void print_arr(int *arr, int arr_n){
	//compute absolute index of thread from 
	//relative 	grid and block thread index
	//sync grids
	    __syncthreads();
	int	index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	int i;
	for(i = index; i < arr_n ; i+=stride)
		printf("%d ", arr[i]);
}

int main(){
	int nthreads = 32; //threads per block
	int nblks; //number of blocks
	int n; //arr size 
	int * arr;
#if part_1
	//run <total_num_blocks, total_num_threads>
	//create N blocks, 1 block per thread 
	//since each block is run in parallel	
	n = 32;
	nblks = n/nthreads;
	make_arr(&arr, n);
	init<<<nblks,nthreads>>>(n, arr);
	print_arr<<<nblks,nthreads>>>(arr,n);
	cudaDeviceSynchronize();
	cudaFree(arr);
#endif
#if part_2_3
	n = 1024;
	nblks = n/nthreads;
	make_arr(&arr, n);
	init<<<nblks,nthreads>>>(n, arr);
	add_i<<<nblks,nthreads>>>(n,arr);
	print_arr<<<nblks, nthreads>>>(arr,n);
	cudaDeviceSynchronize();
	cudaFree(arr);
#endif
#if part_4
	n = 8000;
	nblks = n/nthreads;
	make_arr(&arr, n);
	init<<<nblks, nthreads>>>(n, arr);
	add_i<<<nblks, nthreads>>>(n,arr);
	print_arr<<<nblks,nthreads>>>(arr,n);
	cudaDeviceSynchronize();
	cudaFree(arr);
#endif
	return 1;
}
