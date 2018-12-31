#include "CudaModule.cuh"
#include "device_functions.h"

#include <thread>

// 1 block : 16 * 16 = 256 threads
// 1 grid : 40 * 20 * 100 = 80000 blocks
// Fitted to 640 * 320 image

//*** Full Buffer Space Calculation ***//
// index = (x - 2) + (y - 2) * cols
// at (0, 0) -> index = -2 - 2 * cols
// 2 * cols + 2 for the one side

#define R_DIM							(R_MAX - R_MIN + 1)

#define BLOCK_DIM						16
#define BLOCK_SIZE						256
#define GRID_DIM_ROWS					20
#define GRID_DIM_COLS					40
#define SOBEL_PAD						(IMG_WIDTH + 1) * 2
#define SOBEL_CACHE_DIM					20
#define SOBEL_CACHE_SIZE				400
#define SOBEL_CACHE_LOAD_SECTIONS		2		// ceil(SOBEL_CACHE_SIZE / BLOCK_SIZE)
#define SOBEL_FILTER_GAP				2		// k / 2
#define SOBEL_NORM						168		// 84 for one direction * 2 (2 dir)
#define HOUGH_PAD						(IMG_WIDTH + 1) * R_MAX
#define HOUGH_CACHE_DIM					60
#define HOUGH_CACHE_SIZE				3600
#define HOUGH_CACHE_LOAD_SECTIONS		15		// ceil(SCORE_CACHE_SIZE / BLOCK_SIZE)
#define HOUGH_TOTAL_CARDINALITY			312
#define ARRAY_CONVERSION_THRESHOLD		30
#define ARRAY_CONVERSION_LIST_SIZE		16384
#define HOUGH_CONVENTIONAL_THREADS		512

__device__ __constant__ int ops[] = {18, 42, 78, 102, 138, 162, 198, 222, 258, 283, 317, 343, 377, 403, 437, 464, 496, 524, 556, 585, 615, 646, 674, 706, 707, 733, 734, 767, 768, 792, 793, 828, 829, 851, 852, 890, 910, 951, 952, 968, 969, 1013, 1014, 1015, 1025, 1026, 1027, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 19, 41, 79, 101, 139, 161, 199, 221, 259, 282, 318, 342, 378, 402, 438, 463, 497, 523, 557, 584, 616, 644, 645, 675, 676, 705, 735, 766, 794, 827, 853, 888, 889, 911, 912, 949, 950, 970, 971, 1011, 1012, 1028, 1029, 1073, 1074, 1075, 1085, 1086, 1087, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 20, 40, 80, 100, 140, 160, 200, 220, 260, 281, 319, 341, 379, 401, 439, 462, 498, 522, 558, 583, 617, 643, 677, 704, 736, 765, 795, 826, 854, 887, 913, 948, 972, 1009, 1010, 1030, 1031, 1071, 1072, 1088, 1089, 1133, 1134, 1135, 1145, 1146, 1147, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 21, 39, 81, 99, 141, 159, 201, 219, 261, 280, 320, 340, 380, 400, 440, 461, 499, 521, 559, 581, 582, 618, 619, 642, 678, 703, 737, 763, 764, 796, 797, 824, 825, 855, 856, 885, 886, 914, 915, 946, 947, 973, 974, 1007, 1008, 1032, 1033, 1069, 1070, 1090, 1091, 1130, 1131, 1132, 1148, 1149, 1150, 1193, 1194, 1195, 1205, 1206, 1207, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 22, 38, 82, 98, 142, 158, 202, 218, 262, 279, 321, 339, 381, 399, 441, 459, 460, 500, 501, 520, 560, 580, 620, 641, 679, 701, 702, 738, 739, 762, 798, 823, 857, 884, 916, 945, 975, 1006, 1034, 1067, 1068, 1092, 1093, 1128, 1129, 1151, 1152, 1190, 1191, 1192, 1208, 1209, 1210, 1252, 1253, 1254, 1255, 1265, 1266, 1267, 1268, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324};
__device__ __constant__ int ops_len[R_DIM] = {56, 58, 56, 72, 70};

CudaModule* CudaModule::self = NULL;

// Standard dimensions
const dim3 stdBlockSize(BLOCK_DIM, BLOCK_DIM);
const dim3 stdGridSize(GRID_DIM_COLS, GRID_DIM_ROWS, SAMPLE_SIZE);

const dim3 stdBlockSize_conv(HOUGH_CONVENTIONAL_THREADS);
const dim3 stdGridSize_conv(R_DIM, IMG_HEIGHT, SAMPLE_SIZE);

__global__ void hvSobelFilterKernel(unsigned char* imgdata_dev, unsigned char* edgedata_dev);
__global__ void circleHoughTransformKernel(unsigned char* edgedata_dev, unsigned char* score3ddata_dev);
__global__ void reduceDimensionKernel(unsigned char* score3ddata_dev, unsigned char* score2ddata_dev);

__global__ void arrayConversionKernel(unsigned char* edgedata_dev, int* binarray_dev);
__global__ void circleHoughTransformKernel_conv(int* binarray_dev, unsigned char* score3ddata_dev);
__global__ void circleHoughTransformKernel_nonbconv(unsigned char* edgedata_dev, unsigned char* score3ddata_dev);


inline void SAFE_CUDA(cudaError_t cudaStatus)
{
	if(cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Kernel function error : %s\n", cudaGetErrorString(cudaStatus));
		std::this_thread::sleep_for(std::chrono::seconds(10000));
		exit(1);
	}
}

CudaModule::CudaModule()
{
	initialize();
}

void CudaModule::initialize()
{
	// Allocate
	SAFE_CUDA(cudaMalloc(&imgdata_padded_dev, IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE + SOBEL_PAD + SOBEL_PAD));
	SAFE_CUDA(cudaMalloc(&edgedata_padded_dev, IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE + HOUGH_PAD + HOUGH_PAD));
	SAFE_CUDA(cudaMalloc(&score3ddata_dev,IMG_WIDTH * IMG_HEIGHT * R_DIM * SAMPLE_SIZE));
	SAFE_CUDA(cudaMalloc(&score2ddata_dev, IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE));
	SAFE_CUDA(cudaMalloc(&binarray_dev, ARRAY_CONVERSION_LIST_SIZE * SAMPLE_SIZE * sizeof(int)));
	// Set ROI pointer and fill pad with 0
	imgdata_dev = imgdata_padded_dev + SOBEL_PAD;
	edgedata_dev = edgedata_padded_dev + HOUGH_PAD;
	SAFE_CUDA(cudaMemset(imgdata_padded_dev, 0, SOBEL_PAD));
	SAFE_CUDA(cudaMemset(imgdata_dev + IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE, 0, SOBEL_PAD));
	SAFE_CUDA(cudaMemset(edgedata_padded_dev, 0, HOUGH_PAD));
	SAFE_CUDA(cudaMemset(edgedata_dev + IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE, 0, HOUGH_PAD));
	SAFE_CUDA(cudaMemset(binarray_dev, 0, ARRAY_CONVERSION_LIST_SIZE * SAMPLE_SIZE * sizeof(int)));
}


void CudaModule::hvSobelFilter()
{
	// Call kernel function
	hvSobelFilterKernel<<<stdGridSize, stdBlockSize>>>(imgdata_dev, edgedata_dev);
}


void CudaModule::circleHoughTransform()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("Running Hough transform...");

	cudaEventRecord(start);
	// Call kernel function
	circleHoughTransformKernel<<<stdGridSize, stdBlockSize>>>(edgedata_dev, score3ddata_dev);
	//arrayConversionKernel<<<stdGridSize, stdBlockSize>>>(edgedata_dev, binarray_dev);
	//circleHoughTransformKernel_conv<<<stdGridSize_conv, stdBlockSize_conv>>>(binarray_dev, score3ddata_dev);
	//circleHoughTransformKernel_nonbconv<<<stdGridSize_conv, stdBlockSize_conv>>>(edgedata_dev, score3ddata_dev);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f millisec elapsed!\n", milliseconds);
}

void CudaModule::reduceDimension()
{
	reduceDimensionKernel << <stdGridSize, stdBlockSize >> > (score3ddata_dev, score2ddata_dev);
}

// Sobel Filter size 5 * 5
//				hori							vert
//	|	-5	-4	0	4	5	|		|	-5	-8	-10	-8	-5	|
//	|	-8	-10	0	10	8	|		|	-4	-10	-20	-10	-4	|
//	|	-10	-20	0	20	10	|		|	0	0	0	0	0	|
//	|	-8	-10	0	10	8	|		|	4	10	20	10	4	|
//	|	-5	-4	0	4	5	|	,	|	5	8	10	8	5	|
// 
__global__ void hvSobelFilterKernel(uchar* imgdata_dev, uchar* edgedata_dev)
{
	// Calculate indices
	int block_offset = (blockIdx.y * IMG_WIDTH + blockIdx.x) * BLOCK_DIM;
	int index = block_offset + threadIdx.y * IMG_WIDTH + threadIdx.x;
	int thread_num = threadIdx.y * BLOCK_DIM + threadIdx.x;
	// Calculate the current data address
	uchar* cur_imgdata_dev = imgdata_dev + IMG_WIDTH * IMG_HEIGHT * blockIdx.z;
	// Set shared memories
	__shared__ uchar table_s[SOBEL_CACHE_SIZE];
	if(thread_num < SOBEL_CACHE_SIZE / SOBEL_CACHE_LOAD_SECTIONS)
	{
		int load_table_index = block_offset - IMG_WIDTH - IMG_WIDTH - 2;
		load_table_index += (thread_num / SOBEL_CACHE_DIM) * IMG_WIDTH + thread_num % SOBEL_CACHE_DIM;
		table_s[thread_num] = cur_imgdata_dev[load_table_index];
		table_s[thread_num + SOBEL_CACHE_SIZE / SOBEL_CACHE_LOAD_SECTIONS] = cur_imgdata_dev[load_table_index + SOBEL_CACHE_DIM / SOBEL_CACHE_LOAD_SECTIONS * IMG_WIDTH];
	}
	__syncthreads();

	// Gradients
	int vert, hori, val, offset;
	// Row 1
	offset = threadIdx.y * SOBEL_CACHE_DIM + threadIdx.x;
	hori = vert = table_s[offset++] * -5;
	val = table_s[offset++] * 4;
	hori -= val;
	vert -= val + val;
	vert -= table_s[offset++] * 10;
	val = table_s[offset++] * 4;
	hori += val;
	vert -= val + val;
	val = table_s[offset] * 5;
	hori += val;
	vert -= val;
	// Row 2
	offset += SOBEL_CACHE_DIM - 4;
	val = table_s[offset++] * 4;
	hori -= val + val;
	vert -= val;
	val = table_s[offset++] * 10;
	hori -= val;
	vert -= val;
	vert -= table_s[offset++] * 20;
	val = table_s[offset++] * 10;
	hori += val;
	vert -= val;
	val = table_s[offset] * 4;
	hori += val + val;
	vert -= val;
	// Row 3
	offset += SOBEL_CACHE_DIM - 4;
	hori += (-table_s[offset] + -table_s[offset + 1] * 2 + table_s[offset + 3] * 2 + table_s[offset + 4]) * 10;
	// Row 4
	offset += SOBEL_CACHE_DIM;
	val = table_s[offset++] * 4;
	hori -= val + val;
	vert += val;
	val = table_s[offset++] * 10;
	hori -= val;
	vert += val;
	vert += table_s[offset++] * 20;
	val = table_s[offset++] * 10;
	hori += val;
	vert += val;
	val = table_s[offset] * 4;
	hori += val + val;
	vert += val;
	// Row 5
	offset += SOBEL_CACHE_DIM - 4;
	val = table_s[offset++] * 5;
	hori -= val;
	vert += val;
	val = table_s[offset++] * 4;
	hori -= val;
	vert += val + val;
	vert += table_s[offset++] * 10;
	val = table_s[offset++] * 4;
	hori += val;
	vert += val + val;
	val = table_s[offset] * 5;
	hori += val;
	vert += val;
	val = (abs(vert) + abs(hori)) / SOBEL_NORM;
	if(val > 255) val = 255;

	uchar* cur_edgedata_dev = edgedata_dev + IMG_WIDTH * IMG_HEIGHT * blockIdx.z;
	cur_edgedata_dev[index] = val;
}


__global__ void circleHoughTransformKernel(unsigned char* edgedata_dev, unsigned char* scoredata_dev)
{
	// Calculate indices
	int block_offset = (blockIdx.y * IMG_WIDTH + blockIdx.x) * BLOCK_DIM;
	int thread_num = threadIdx.y * BLOCK_DIM + threadIdx.x;
	// Calculate the current data address
	uchar* cur_edgedata_dev = edgedata_dev + IMG_WIDTH * IMG_HEIGHT * blockIdx.z;
	// Set shared memories
	__shared__ uchar table_s[HOUGH_CACHE_SIZE];
	if(thread_num < HOUGH_CACHE_SIZE / HOUGH_CACHE_LOAD_SECTIONS)
	{
		int load_table_index = block_offset - HOUGH_PAD;
		load_table_index += (thread_num / HOUGH_CACHE_DIM) * IMG_WIDTH + thread_num % HOUGH_CACHE_DIM;
		for(int i = 0; i < HOUGH_CACHE_LOAD_SECTIONS; i++)
			table_s[thread_num + i * HOUGH_CACHE_SIZE / HOUGH_CACHE_LOAD_SECTIONS] = cur_edgedata_dev[load_table_index + i * HOUGH_CACHE_DIM / HOUGH_CACHE_LOAD_SECTIONS * IMG_WIDTH];
	}
	__syncthreads();

	int centerindex = R_MAX + threadIdx.x + HOUGH_CACHE_DIM * (threadIdx.y + R_MAX);
	int index = (block_offset + threadIdx.y * IMG_WIDTH + threadIdx.x) * R_DIM;
	uchar* cur_scoredata_dev = scoredata_dev + IMG_WIDTH * IMG_HEIGHT * R_DIM * blockIdx.z;
	int i = 0;
	int cur_end = 0;
	for(int r = 0; r < R_DIM; r++)
	{
		int cardinality = ops_len[r];
		cur_end += cardinality;
		int totalscore = 0;
		for(; i < cur_end; i++)
		{
			int offset = ops[i];
			totalscore += table_s[centerindex + offset];
			totalscore += table_s[centerindex - offset];
		}

		totalscore /= (cardinality + cardinality);
		cur_scoredata_dev[index + r] = totalscore;
	}
}

__global__ void arrayConversionKernel(unsigned char* edgedata_dev, int* binarray_dev)
{
	uchar* cur_edgedata_dev = edgedata_dev + IMG_WIDTH * IMG_HEIGHT * blockIdx.z;
	int thread_num = blockIdx.x * BLOCK_DIM + threadIdx.x + (blockIdx.y * BLOCK_DIM + threadIdx.y) * IMG_WIDTH;

	__shared__ int shared_array[BLOCK_SIZE];
	__shared__ int shared_index;
	int register_index = -1;
	if(threadIdx.x == 0) shared_index = 0;
	__syncthreads();

	if(cur_edgedata_dev[thread_num] >= ARRAY_CONVERSION_THRESHOLD)
	{
		register_index  = atomicAdd(&shared_index, 1);
		shared_array[register_index] = thread_num;
	}

	__syncthreads();

	__shared__ int global_offset;
	int* cur_binarray_dev = binarray_dev + ARRAY_CONVERSION_LIST_SIZE * blockIdx.z;
	int binarray_index;

	if(register_index == 0)
		global_offset = atomicAdd(cur_binarray_dev, shared_index);

	__syncthreads();

	if(register_index > -1)
	{
		binarray_index = global_offset + register_index + 1;
		cur_binarray_dev[binarray_index] = thread_num;
	}
}

__global__ void circleHoughTransformKernel_conv(int* binarray_dev, unsigned char* scoredata_dev)
{
	int r = blockIdx.x + R_MIN;
	int b = blockIdx.y;
	int* cur_binarray_dev = binarray_dev + ARRAY_CONVERSION_LIST_SIZE * blockIdx.z;
	int cardinality = cur_binarray_dev[0];

	__shared__ int totalscore[IMG_WIDTH];
	for(int i = 0; i <= IMG_WIDTH / HOUGH_CONVENTIONAL_THREADS; i++)
	{
		int shared_index = threadIdx.x + i * HOUGH_CONVENTIONAL_THREADS;
		if(shared_index < IMG_WIDTH) totalscore[shared_index] = 0;
	}
	
	for(int i = 0; i <= cardinality / HOUGH_CONVENTIONAL_THREADS; i++)
	{
		int binarray_index = threadIdx.x + i * HOUGH_CONVENTIONAL_THREADS;
		if(binarray_index < cardinality)
		{
			int thread_num = cur_binarray_dev[binarray_index + 1];
			int x = thread_num % IMG_WIDTH;
			int y = thread_num / IMG_WIDTH;
			if(b <= y + r && b >= y - r)
			{
				int y_b = y - b;
				int det = sqrtf(r * r - y_b * y_b);
				int a1 = x + det;
				int a2 = x - det;

				if(a1 >= 0 && a1 < IMG_WIDTH) atomicAdd(&totalscore[a1], 1);
				if(a2 >= 0 && a2 < IMG_WIDTH) atomicAdd(&totalscore[a2], 1);
			}
		}
	}
	
	uchar* cur_scoredata_dev = scoredata_dev + IMG_WIDTH * IMG_HEIGHT * R_DIM * blockIdx.z;
	int row_offset = b * IMG_WIDTH * R_DIM;
	for(int i = 0; i <= IMG_WIDTH / HOUGH_CONVENTIONAL_THREADS; i++)
	{
		int shared_index = threadIdx.x + i * HOUGH_CONVENTIONAL_THREADS;
		if(shared_index < IMG_WIDTH)
		{
			int index = row_offset + shared_index * R_DIM + blockIdx.x;
			cur_scoredata_dev[index] = totalscore[shared_index];
		}
	}
}


__global__ void circleHoughTransformKernel_nonbconv(unsigned char* edgedata_dev, unsigned char* scoredata_dev)
{
	int r = blockIdx.x + R_MIN;
	int b = blockIdx.y;
	uchar* cur_edgedata_dev = edgedata_dev + IMG_WIDTH * IMG_HEIGHT * blockIdx.z;

	__shared__ int totalscore[IMG_WIDTH];
	for(int i = 0; i <= IMG_WIDTH / HOUGH_CONVENTIONAL_THREADS; i++)
	{
		int shared_index = threadIdx.x + i * HOUGH_CONVENTIONAL_THREADS;
		if(shared_index < IMG_WIDTH) totalscore[shared_index] = 0;
	}
	
	for(int i = 0; i <= IMG_WIDTH * IMG_HEIGHT / HOUGH_CONVENTIONAL_THREADS; i++)
	{
		int edge_index = threadIdx.x + i * HOUGH_CONVENTIONAL_THREADS;
		if(edge_index < IMG_WIDTH * IMG_HEIGHT)
		{
			int x = edge_index % IMG_WIDTH;
			int y = edge_index / IMG_WIDTH;
			if(b <= y + r && b >= y - r)
			{
				int y_b = y - b;
				int det = sqrtf(r * r - y_b * y_b);
				int a1 = x + det;
				int a2 = x - det;

				if(a1 >= 0 && a1 < IMG_WIDTH) atomicAdd(&totalscore[a1], cur_edgedata_dev[edge_index]);
				if(a2 >= 0 && a2 < IMG_WIDTH) atomicAdd(&totalscore[a2], cur_edgedata_dev[edge_index]);
			}
		}
	}

	uchar* cur_scoredata_dev = scoredata_dev + IMG_WIDTH * IMG_HEIGHT * R_DIM * blockIdx.z;
	int row_offset = b * IMG_WIDTH * R_DIM;
	for(int i = 0; i <= IMG_WIDTH / HOUGH_CONVENTIONAL_THREADS; i++)
	{
		int shared_index = threadIdx.x + i * HOUGH_CONVENTIONAL_THREADS;
		if(shared_index < IMG_WIDTH)
		{
			int index = row_offset + shared_index * R_DIM + blockIdx.x;
			cur_scoredata_dev[index] = totalscore[shared_index] / ops_len[blockIdx.x];
		}
	}
}


__global__ void reduceDimensionKernel(unsigned char* score3ddata_dev, unsigned char* score2ddata_dev)
{
	// Calculate indices
	int block_offset = (blockIdx.y * IMG_WIDTH + blockIdx.x) * BLOCK_DIM;
	int index2d = block_offset + threadIdx.y * IMG_WIDTH + threadIdx.x;
	int index3d = index2d * R_DIM;
	// Calculate the current data address
	uchar* cur_score3ddata_dev = score3ddata_dev + IMG_WIDTH * IMG_HEIGHT * R_DIM * blockIdx.z;
	uchar* cur_score2ddata_dev = score2ddata_dev + IMG_WIDTH * IMG_HEIGHT * blockIdx.z;
	int totalscore = 0;
	for(int i = 0; i < R_DIM; i++) totalscore += cur_score3ddata_dev[index3d + i];
	cur_score2ddata_dev[index2d] = totalscore / R_DIM;
}


void CudaModule::sync()//int index_img)
{
	SAFE_CUDA(cudaDeviceSynchronize());
}


void CudaModule::upLoadImg(uchar* data)
{
	SAFE_CUDA(cudaMemcpy(imgdata_dev, data, IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE, cudaMemcpyHostToDevice));
}

void CudaModule::upLoadEdge(uchar* data)
{
	SAFE_CUDA(cudaMemcpy(edgedata_dev, data, IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE, cudaMemcpyHostToDevice));
}

void CudaModule::downLoadEdge(uchar* data)
{
	SAFE_CUDA(cudaMemcpy(data, edgedata_dev, IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE, cudaMemcpyDeviceToHost));
}


void CudaModule::downLoadScore2d(uchar* data)
{
	SAFE_CUDA(cudaMemcpy(data, score2ddata_dev, IMG_WIDTH * IMG_HEIGHT * SAMPLE_SIZE, cudaMemcpyDeviceToHost));
}


void CudaModule::downLoadBinArray(int* data)
{
	SAFE_CUDA(cudaMemcpy(data, binarray_dev, 16384 * SAMPLE_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
}