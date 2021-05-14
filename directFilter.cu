// CUDA implementation of the direct filter estimation algorithm
// Copyright (C) 2021  Gerardo Becerra

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// You can reach me at gbecerra@javeriana.edu.co

#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <helper_cuda.h>
#include "matio.h"

typedef struct {
	int rows;
	int cols;
	double *data;
} matrix_t;

// Computes the first step: squared differences of current regressor with respect
// to all regressors in the dataset
__global__ void smSqrdDiffKernel(int m, int Ntot, double *phi, double *dataset, double *sqrdiff)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = i % m;			// Remainder
	if (i < Ntot){
		double diff = phi[j] - dataset[i];
		sqrdiff[i] = diff*diff;
	}
}

// Computes the second step: delta (accumulate, square root) for each regressor
//				smAccumSqrtKernel(m, Ndb, epsilon, gamma, d_fc_db, d_sqrdiff, d_fcub, d_fclb);
__global__ void smAccumSqrtKernel(int m, int Ndb, double epsilon, double gamma,
							double *fc_db, double *sqrdiff, double *fcub, double *fclb)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;		// Index of regressor in dataset
	if (i < Ndb){
		double accum = 0, delta;
		for (int j = 0; j < m; j++){
			accum += sqrdiff[j+m*i];
		}
		delta = sqrt(accum);
		fcub[i] = fc_db[i] + epsilon + gamma*delta;
		fclb[i] = fc_db[i] - epsilon - gamma*delta;
	}
}

__inline__ __device__
double warpReduceAccum(double val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0x0003FFFF,val,offset);
  return val;
}

__inline__ __device__
double warpReduceUpperBound(double minVal) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		minVal = min(minVal, __shfl_down_sync(0xFFFFFFFF, minVal, offset));
	return minVal;
}

__inline__ __device__
double warpReduceLowerBound(double maxVal) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		maxVal = max(maxVal, __shfl_down_sync(0xFFFFFFFF, maxVal, offset));
	return maxVal;
}

__inline__ __device__
double blockReduceUpperBound(double minVal) {
	static __shared__ double shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	minVal = warpReduceUpperBound(minVal);

	//write reduced value to shared memory
	if (lane == 0)
		shared[wid] = minVal;
	__syncthreads();

	//ensure we only grab a value from shared memory if that warp existed
	minVal =
			(threadIdx.x < blockDim.x / warpSize) ?
					shared[lane] : double(DBL_MAX);
	if (wid == 0)
		minVal = warpReduceUpperBound(minVal);

	return minVal;
}

__inline__ __device__
double blockReduceLowerBound(double maxVal) {
	static __shared__ double shared[32];
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
	maxVal = warpReduceLowerBound(maxVal);

	//write reduced value to shared memory
	if (lane == 0)
		shared[wid] = maxVal;
	__syncthreads();

	//ensure we only grab a value from shared memory if that warp existed
	maxVal =
			(threadIdx.x < blockDim.x / warpSize) ?
					shared[lane] : double(-DBL_MAX);
	if (wid == 0)
		maxVal = warpReduceLowerBound(maxVal);

	return maxVal;
}

__global__ void smUpperBoundKernel(double *in, double* out, int N) {
	double minVal = double(DBL_MAX);
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
			i += blockDim.x * gridDim.x) {
		minVal = min(minVal, in[i]);
	}
	minVal = blockReduceUpperBound(minVal);
	if (threadIdx.x == 0)
		out[blockIdx.x] = minVal;
}

__global__ void smLowerBoundKernel(double *in, double *out, int N) {
	double maxVal = double(-DBL_MAX);
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
			i += blockDim.x * gridDim.x) {
		maxVal = max(maxVal, in[i]);
	}
	maxVal = blockReduceLowerBound(maxVal);
	if (threadIdx.x == 0)
		out[blockIdx.x] = maxVal;
}

__global__ void smEstimateKernel(int j, double *fcub, double *fclb, double *result) {
	result[j+0] = (*fcub + *fclb)/2;
	result[j+1] = *fcub;
	result[j+2] = *fclb;
}

// Reads matrix from mat file
matrix_t* ReadMatrix(mat_t *matfile, const char* name){
	matvar_t *matvar;

	int start[2]={0,0},stride[2]={1,1},edge[2]={1,1};

	matvar = Mat_VarReadInfo(matfile,name);
	if ( NULL == matvar ) {
		printf("Variable not found, or error reading MAT file %s\n",name);
		return NULL;
	} else {
		matrix_t *matrix = (matrix_t *)malloc(sizeof(matrix_t));
		int rows = matvar->dims[0];
		int cols = matvar->dims[1];
		edge[0] = rows;
		edge[1] = cols;
		matrix->data = (double *)malloc(sizeof(double)*rows*cols);
		Mat_VarReadData(matfile,matvar,matrix->data,start,stride,edge);
		matrix->rows = rows;
		matrix->cols = cols;
		return matrix;
	}
}

int main(int argc,char **argv)
{
	// Open the mat file
	mat_t    *matfp1, *matfp2;
	matfp1 = Mat_Open(argv[1],MAT_ACC_RDONLY);
	if ( NULL == matfp1 ) {
		printf("Error opening MAT file: \"%s\"!\n",argv[1]);
		return EXIT_FAILURE;
	} else
		printf("File opened succesfully: \"%s\"!\n",argv[1]);

	// Read the variables from the mat file
	matrix_t *m_epsilon		= ReadMatrix(matfp1, "epsilon");
	matrix_t *m_gamma 		= ReadMatrix(matfp1, "gamma");
	matrix_t *m_phi_db		= ReadMatrix(matfp1, "phi_train");
	matrix_t *m_fc_db 		= ReadMatrix(matfp1, "v_train");
	matrix_t *m_phi_test	= ReadMatrix(matfp1, "phi_test");
	matrix_t *m_fc_test		= ReadMatrix(matfp1, "v_test");
	matrix_t *m_sqrt_vu 	= ReadMatrix(matfp1, "sqrt_vu");

	findCudaDevice(argc, (const char **)argv);

	// Initialize main parameters
	double epsilon = m_epsilon->data[0];
	double gamma   = m_gamma->data[0];
	int m   = m_phi_db->rows;
	int Ndb = m_phi_db->cols;
	int Nt  = m_phi_test->cols;
	// Nt = 1000;

	// Initialize device arrays
	// Regressors for training dataset: allocates device memory
	// and copies data from the host
	double *d_phi_db;
	size_t size = m * Ndb * sizeof(double);
	checkCudaErrors(cudaMallocManaged(&d_phi_db, size));
	checkCudaErrors(cudaMemcpy(d_phi_db, m_phi_db->data, size, cudaMemcpyHostToDevice));

	// Output for training dataset: allocates device memory
	// and copies data from the host
	double *d_fc_db;
	size = Ndb * sizeof(double);
	checkCudaErrors(cudaMallocManaged(&d_fc_db, size));
	checkCudaErrors(cudaMemcpy(d_fc_db, m_fc_db->data, size, cudaMemcpyHostToDevice));

	// Regressor scaling vector: allocates device memory
	// and copies data from the host
	double *d_sqrt_vu;
	size = m * sizeof(double);
	checkCudaErrors(cudaMallocManaged(&d_sqrt_vu, size));
	checkCudaErrors(cudaMemcpy(d_sqrt_vu, m_sqrt_vu->data, size, cudaMemcpyHostToDevice));

	// Squared differences array: allocates device memory
	// for storing the results from the first kernel
	double *d_sqrdiff1, *d_sqrdiff2;
	size = m * Ndb * sizeof(double);
	checkCudaErrors(cudaMallocManaged(&d_sqrdiff1, size));
	checkCudaErrors(cudaMallocManaged(&d_sqrdiff2, size));

	// Set membership: allocates host memory for copying
	// the results from the device
	double *h_result;
	checkCudaErrors(cudaMallocHost(&h_result, 3*Nt*sizeof(double)));

	// Set membership: allocates host memory for saving
	// the results to a Matlab .mat file
	double *h_fcub, *h_fclb, *h_vest;
	checkCudaErrors(cudaMallocHost(&h_fcub, Nt*sizeof(double)));
	checkCudaErrors(cudaMallocHost(&h_fclb, Nt*sizeof(double)));
	checkCudaErrors(cudaMallocHost(&h_vest, Nt*sizeof(double)));

	// Set membership: allocates device memory for intermediate
	// results
	double *d_fcub1, *d_fclb1, *d_fcub_min1, *d_fclb_max1, *d_result;
	double *d_fcub2, *d_fclb2, *d_fcub_min2, *d_fclb_max2;
	size = Ndb * sizeof(double);
	checkCudaErrors(cudaMallocManaged(&d_fcub1, size));
	checkCudaErrors(cudaMallocManaged(&d_fclb1, size));
	checkCudaErrors(cudaMallocManaged(&d_fcub_min1, sizeof(double)*1024));
	checkCudaErrors(cudaMallocManaged(&d_fclb_max1, sizeof(double)*1024));
	checkCudaErrors(cudaMallocManaged(&d_fcub2, size));
	checkCudaErrors(cudaMallocManaged(&d_fclb2, size));
	checkCudaErrors(cudaMallocManaged(&d_fcub_min2, sizeof(double)*1024));
	checkCudaErrors(cudaMallocManaged(&d_fclb_max2, sizeof(double)*1024));
	checkCudaErrors(cudaMallocManaged(&d_result, 3*Nt*sizeof(double)));

	// k-th test regressor: allocates host and device memory
	// for the original and test k-th regressor
	double *d_phi_k1, *d_phi_k2, *h_phi_scld_k1, *h_phi_scld_k2;
	size = m * sizeof(double);
	checkCudaErrors(cudaMallocManaged(&d_phi_k1, size));
	checkCudaErrors(cudaMallocManaged(&d_phi_k2, size));
	checkCudaErrors(cudaMallocHost(&h_phi_scld_k1, size));
	checkCudaErrors(cudaMallocHost(&h_phi_scld_k2, size));

	// Creates a non-default stream for running the kernels
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// Creates events for timing the execution
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float avgTimeGPU = 0;

	// ********************************************************************************
	// *************** Runs the set membership algorithm in the GPU *******************
	// ********************************************************************************
	printf("Computing SM estimate using the GPU... ");

	// Configures grid and block dimensions for the first two kernels
	int dimGrid = 128;
	int dimBlock = 1024;

	// Configures grid and block dimensions for the other kernels
	int threads = 512;
    int blocks = min((Ndb + threads - 1) / threads, 1024);

    // Starts measuring the total execution time
    cudaEventRecord(start);
	for (int k = 0; k < Nt; k++){
		// When the test regressors are obtained from online data, the scaling process
		// has to be done here... In this case, the test regressors are pre-scaled
		for (int j=0; j<m; j++){
			h_phi_scld_k1[j] = m_phi_test->data[m*k+j];
		}
		// for (int j=0; j<m; j++){
		// 	h_phi_scld_k2[j] = m_phi_test->data[m*(k+1)+j];
		// }
		checkCudaErrors(cudaMemcpyAsync(d_phi_k1, h_phi_scld_k1, size, cudaMemcpyHostToDevice, stream1));
		smSqrdDiffKernel<<<dimGrid,dimBlock,0,stream1>>>(m, m*Ndb, d_phi_k1, d_phi_db, d_sqrdiff1);

		// checkCudaErrors(cudaMemcpyAsync(d_phi_k2, h_phi_scld_k2, size, cudaMemcpyHostToDevice, stream2));
		// smSqrdDiffKernel<<<dimGrid,dimBlock,0,stream2>>>(m, m*Ndb, d_phi_k2, d_phi_db, d_sqrdiff2);

		smAccumSqrtKernel<<<dimGrid,dimBlock,0,stream1>>>(m, Ndb, epsilon, gamma, d_fc_db, d_sqrdiff1, d_fcub1, d_fclb1);
		// smAccumSqrtKernel<<<dimGrid,dimBlock,0,stream2>>>(m, Ndb, epsilon, gamma, d_fc_db, d_sqrdiff2, d_fcub2, d_fclb2);
	
		smLowerBoundKernel<<<blocks,threads,0,stream1>>>(d_fclb1, d_fclb_max1, Ndb);
		// smLowerBoundKernel<<<blocks,threads,0,stream2>>>(d_fclb2, d_fclb_max2, Ndb);

		smUpperBoundKernel<<<blocks,threads,0,stream1>>>(d_fcub1, d_fcub_min1, Ndb);
		// smUpperBoundKernel<<<blocks,threads,0,stream2>>>(d_fcub2, d_fcub_min2, Ndb);

		smLowerBoundKernel<<<1,1024,0,stream1>>>(d_fclb_max1, d_fclb_max1, blocks);
		// smLowerBoundKernel<<<1,1024,0,stream2>>>(d_fclb_max2, d_fclb_max2, blocks);
	
		smUpperBoundKernel<<<1,1024,0,stream1>>>(d_fcub_min1, d_fcub_min1, blocks);
		// smUpperBoundKernel<<<1,1024,0,stream2>>>(d_fcub_min2, d_fcub_min2, blocks);

		smEstimateKernel<<<1,1,0,stream1>>>(3*k,   d_fcub_min1, d_fclb_max1, d_result);
		// smEstimateKernel<<<1,1,0,stream2>>>(3*k+1, d_fcub_min2, d_fclb_max2, d_result);

		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpyAsync(&h_result[3*k],     &d_result[3*k], 3*sizeof(double), cudaMemcpyDeviceToHost, stream1));
		// checkCudaErrors(cudaMemcpyAsync(&h_result[3*(k+1)], &d_result[3*(k+1)], 3*sizeof(double), cudaMemcpyDeviceToHost, stream2));
	}
	checkCudaErrors(cudaDeviceSynchronize());

	// Stops measuring time and computes average execution time for each iteration
	cudaEventRecord(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	avgTimeGPU += milliseconds;
	avgTimeGPU = avgTimeGPU / Nt;

	checkCudaErrors(cudaStreamDestroy(stream1));

	printf("Done!\n");

	// ********************************************************************************
	// *************** Runs the set membership algorithm in the CPU *******************
	// ********************************************************************************
	printf("Computing SM estimate using the CPU... ");
	double fcub_cpu[Nt], fclb_cpu[Nt], vest_cpu[Nt];
	for (int k = 0; k < Nt; k++){
		double deltaphi[Ndb];
		double min_ub = double(DBL_MAX);
		double max_lb = double(-DBL_MAX);
		for (int i = 0; i < Ndb; i++){
			double accum = 0;
			for (int j = 0; j < m; j++){
				double phi_diff = m_phi_test->data[m*k+j] - m_phi_db->data[m*i+j];
				accum += phi_diff*phi_diff;
			}
			deltaphi[i] = sqrt(accum);
			double ub = m_fc_db->data[i] + epsilon + gamma * deltaphi[i];
			double lb = m_fc_db->data[i] - epsilon - gamma * deltaphi[i];
			if (ub < min_ub)
				min_ub = ub;
			if (lb > max_lb)
				max_lb = lb;
		}
		fcub_cpu[k] = min_ub;
		fclb_cpu[k] = max_lb;
		vest_cpu[k] = (min_ub + max_lb)/2;
	}
	printf("Done!\n\n");

	// ********************************************************************************
	// *************** Compares the results for both implementations ******************
	// ********************************************************************************
	double accum_est = 0, accsqrdiff_cpu = 0, accsqrdiff_gpu = 0;
	for (int k = 0; k < Nt; k++){
		h_vest[k] = h_result[3*k];
		h_fcub[k] = h_result[3*k+1];
		h_fclb[k] = h_result[3*k+2];
		double diff_est = h_vest[k] - vest_cpu[k];
		accsqrdiff_cpu += (vest_cpu[k]   - m_fc_test->data[k])*(vest_cpu[k]   - m_fc_test->data[k]);
		accsqrdiff_gpu += (h_vest[k] - m_fc_test->data[k])*(h_vest[k] - m_fc_test->data[k]);
		accum_est += diff_est*diff_est;
	}
	double RMSE_cpu = sqrt(accsqrdiff_cpu / Nt);
	double RMSE_gpu = sqrt(accsqrdiff_gpu / Nt);

	double RMSE_est = sqrt(accum_est/Nt);
	printf("Grid size = %d, Block size = %d\n",dimGrid,dimBlock);
	printf("Regressor size = %d, Dataset size = %d\n",m,Ndb);
	printf("Average GPU execution time = %f us\n", 1000*avgTimeGPU);
	printf("Average GPU execution rate = %f kHz\n", 1/avgTimeGPU);
	printf("Difference between CPU / GPU = %f\n", RMSE_est);
	printf("RMSE CPU = %f\n", RMSE_cpu);
	printf("RMSE GPU = %f\n\n", RMSE_gpu);

	// Store the results in a new mat file
	printf("Saving results... ");
	matvar_t *matvar;
	size_t dims[2];
	dims[0] = Nt;
	dims[1] = 1;

	matfp2 = Mat_CreateVer(argv[2],NULL,MAT_FT_DEFAULT);
	if (matfp2 == NULL)
		printf("Error creating MAT file \"estim.mat\"\n");
	else {
		matvar = Mat_VarCreate("fcub",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,h_fcub,0);
		if (matvar == NULL)
			printf("Error creating variable for fcub");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("fclb",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,h_fclb,0);
		if (matvar == NULL)
			printf("Error creating variable for fclb");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("vest",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,h_vest,0);
		if (matvar == NULL)
			printf("Error creating variable for vest");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("fcub_cpu",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,fcub_cpu,0);
		if (matvar == NULL)
			printf("Error creating variable for fcub_cpu");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("fclb_cpu",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,fclb_cpu,0);
		if (matvar == NULL)
			printf("Error creating variable for fclb_cpu");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("vest_cpu",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,vest_cpu,0);
		if (matvar == NULL)
			printf("Error creating variable for vest_cpu");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		dims[0] = 1;
		dims[1] = 1;
		matvar = Mat_VarCreate("gridSize",MAT_C_INT32,MAT_T_INT32,2,dims,&dimGrid,0);
		if (matvar == NULL)
			printf("Error creating variable for gridSize");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("blockSize",MAT_C_INT32,MAT_T_INT32,2,dims,&dimBlock,0);
		if (matvar == NULL)
			printf("Error creating variable for blockSize");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("m",MAT_C_INT32,MAT_T_INT32,2,dims,&m,0);
		if (matvar == NULL)
			printf("Error creating variable for m");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("Ndb",MAT_C_INT32,MAT_T_INT32,2,dims,&Ndb,0);
		if (matvar == NULL)
			printf("Error creating variable for Ndb");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("timeGPUms",MAT_C_SINGLE,MAT_T_SINGLE,2,dims,&avgTimeGPU,0);
		if (matvar == NULL)
			printf("Error creating variable for timeGPU");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("diffCPU_GPU",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,&RMSE_est,0);
		if (matvar == NULL)
			printf("Error creating variable for diffCPU_GPU");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("rmseGPU",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,&RMSE_gpu,0);
		if (matvar == NULL)
			printf("Error creating variable for rmseGPU");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
		matvar = Mat_VarCreate("rmseCPU",MAT_C_DOUBLE,MAT_T_DOUBLE,2,dims,&RMSE_cpu,0);
		if (matvar == NULL)
			printf("Error creating variable for rmseCPU");
		else {
			Mat_VarWrite(matfp2,matvar,MAT_COMPRESSION_NONE);
			Mat_VarFree(matvar);
		}
	}
	printf("Done!\n");

	printf("Freeing Memory... ");

	// Free used memory
	Mat_Close(matfp1);
	Mat_Close(matfp2);
	free(m_epsilon->data);
	free(m_epsilon);
	free(m_gamma->data);
	free(m_gamma);
	free(m_phi_db->data);
	free(m_phi_db);
	free(m_fc_db->data);
	free(m_fc_db);
	free(m_phi_test->data);
	free(m_phi_test);
	free(m_fc_test->data);
	free(m_fc_test);
	free(m_sqrt_vu->data);
	free(m_sqrt_vu);
	checkCudaErrors(cudaFree(d_phi_db));
	checkCudaErrors(cudaFree(d_fc_db));
	checkCudaErrors(cudaFree(d_sqrt_vu));
	checkCudaErrors(cudaFree(d_sqrdiff1));
	checkCudaErrors(cudaFree(d_sqrdiff2));
	checkCudaErrors(cudaFree(d_fcub1));
	checkCudaErrors(cudaFree(d_fclb1));
	checkCudaErrors(cudaFree(d_fcub_min1));
	checkCudaErrors(cudaFree(d_fclb_max1));
	checkCudaErrors(cudaFree(d_fcub2));
	checkCudaErrors(cudaFree(d_fclb2));
	checkCudaErrors(cudaFree(d_fcub_min2));
	checkCudaErrors(cudaFree(d_fclb_max2));
	checkCudaErrors(cudaFree(d_result));
	checkCudaErrors(cudaFreeHost(h_result));
	checkCudaErrors(cudaFreeHost(h_fcub));
	checkCudaErrors(cudaFreeHost(h_fclb));
	checkCudaErrors(cudaFreeHost(h_vest));

	cudaDeviceReset();

	printf("Done!\n");
	printf("Finished! \n");
	return EXIT_SUCCESS;
}
