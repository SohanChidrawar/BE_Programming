%%writefile add.cu

#include <iostream>
#include <cstdlib> // Include <cstdlib> for rand()
using namespace std;

__global__ void add(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
        printf("Thread %d: %d + %d = %d\n", tid, A[tid], B[tid], C[tid]);
    }
}

void print(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

int main() {
    int N;
    cout << "Enter the size of the vectors: ";
    cin >> N;

    // Allocate host memory
    int* A = new int[N];
    int* B = new int[N];
    int* C = new int[N];

    // Initialize host arrays
    cout << "Enter elements for vector A: ";
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }

    cout << "Enter elements for vector B: ";
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }

    cout << "Vector A: ";
    print(A, N);
    cout << "Vector B: ";
    print(B, N);

    int* X, * Y, * Z;
    size_t vectorBytes = N * sizeof(int);

    // Allocate device memory
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    // Copy data from host to device
    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);

    // Copy result from device to host
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);

    cout << "Addition: ";
    print(C, N);

    // Free device memory
    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}


/*
----------------------------------------------------------------------------------------------------------------------
Enter the size of the vectors: 4
Enter elements for vector A: 1 5 8 4
Enter elements for vector B: 9 4 0 7
Vector A: 1 5 8 4 
Vector B: 9 4 0 7 
Thread 0: 1 + 9 = 10
Thread 1: 5 + 4 = 9
Thread 2: 8 + 0 = 8
Thread 3: 4 + 7 = 11
Addition: 10 9 8 11 

*/
