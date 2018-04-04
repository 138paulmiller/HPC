#include <stdio.h>
#include <math.h>
#define part_1 0
#define part_2_3 0
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
	int	index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x*blockDim.x;
	if(stride+index < arr_n)
		printf("%d ", arr[index]);  
}
int main(){

	int n = 32;
	int * arr;
#if part_1
	make_arr(&arr, n);
	//run <total_num_blocks, total_num_threads>
	//create N blocks, 1 block per thread 
	//since each block is run in parallel	
	init<<<n,1>>>(n, arr);
	cudaDeviceSynchronize();
	//print inorder so run 1 blocks with n threads to print
	print_arr<<<1,n>>>(arr,n);
	cudaFree(arr);
#endif
#if part_2_3
	n = 1024;
	make_arr(&arr, n);
	init<<<n,1>>>(n, arr);
	add_i<<<n,1>>>(n,arr);
	print_arr<<<1,n>>>(arr,n);
	cudaDeviceSynchronize();
	cudaFree(arr);
#endif
#if part_4
	n = 10000;
	make_arr(&arr, n);
	init<<<n,1>>>(n, arr);
	add_i<<<n,1>>>(n,arr);
	//TODO - print each sub array managed by block
	int nblk = (n-1)/32;
	print_arr<<<nblk,32>>>(arr,n);
	cudaDeviceSynchronize();
	cudaFree(arr);
#endif
	return 1;
}
