#Matrix multiplication usin CUDA

 %%writefile matrix_mult.cu
 #include <iostream>
 #include <cuda.h>
 using namespace std;
 #define BLOCK_SIZE 2
 __global__ void gpuMM(float *A, float *B, float *C, int N) 
 {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;
    for (int n = 0; n < N; ++n)
        sum += A[row * N + n] * B[n * N + col];
    C[row * N + col] = sum;
 }
 
 int main(int argc, char *argv[]) 
 {
    int N;
    float K;
    // Perform matrix multiplication C = A*B
     // where A, B and C are NxN matrices
    // Restricted to matrices where N = K*BLOCK_SIZE;
    cout << "Enter a value for size/2 of matrix: ";
    cin >> K;
    K = 1;
    N = K * BLOCK_SIZE;
    cout << "\nExecuting Matrix Multiplication" << endl;
    cout << "Matrix size: " << N << "x" << N << endl;
    // Allocate memory on the host
    float *hA, *hB, *hC;
    hA = new float[N * N];
    hB = new float[N * N];
    hC = new float[N * N];
    // Initialize matrices on the host with random values
    srand(time(NULL)); // Seed the random number generator
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            hA[j * N + i] = rand() % 10; // Generate random value between 0 and 9
            hB[j * N + i] = rand() % 10; // Generate random value between 0 and 9
        }
    }

    // Allocate memory on the device
    int size = N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(K, K);

    // Copy matrices from the host to device
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);
    // Execute the matrix multiplication kernel
    gpuMM<<<grid, threadBlock>>>(dA, dB, dC, N);
    // Copy the GPU result back to CPU
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
    // Display the result
    cout << "\nResultant matrix:\n";
    for (int row = 0; row < N; row++) 
    {
        for (int col = 0; col < N; col++) 
        {
            cout << hC[row * N + col] << " ";
        }
        cout << endl;
    }

     // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    // Free host memory
    delete[] hA;
    delete[] hB;
    delete[] hC;
    cout << "Finished." << endl;
    return 0;
}
