%%writefile bubblesort2.cu

#include <iostream>  // For standard I/O
#include <vector>  // For dynamic arrays (vectors)
#include <chrono>  // For measuring time
#include <random>  // For random number generation
#include <cuda_runtime.h>  // For CUDA-related functions

using namespace std;

// Device function to swap two integers on the GPU
__device__ void device_swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

// CUDA kernel for odd-even bubble sort
__global__ void kernel_bubble_sort_odd_even(int* arr, int size) {
    bool isSorted = false;  // Flag to check if sorting is complete

    while (!isSorted) {
        isSorted = true;  // Assume array is sorted

        int tid = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate thread ID

        // Even phase: Process elements at even indices
        for (int i = 0; i < size - 1; i += 2) {
            if (tid == i && arr[i] > arr[i + 1]) {  // If adjacent elements are out of order
                device_swap(arr[i], arr[i + 1]);  // Swap if necessary
                isSorted = false;  // Not sorted yet
            }
        }

        __syncthreads();  // Synchronize threads within the block

        // Odd phase: Process elements at odd indices
        for (int i = 1; i < size - 1; i += 2) {
            if (tid == i && arr[i] > arr[i + 1]) {  // If adjacent elements are out of order
                device_swap(arr[i], arr[i + 1]);  // Swap if necessary
                isSorted = false;  // Not sorted yet
            }
        }

        __syncthreads();  // Synchronize threads within the block
    }
}

// Function to perform sequential bubble sort
void sequential_bubble_sort(vector<int>& arr) {
    bool isSorted = false;

    // Continue until the array is sorted
    while (!isSorted) {
        isSorted = true;  // Assume sorted unless proven otherwise

        // Process through the array
        for (int i = 0; i < arr.size() - 1; i++) {
            if (arr[i] > arr[i + 1]) {  // If adjacent elements are out of order
                swap(arr[i], arr[i + 1]);  // Swap them
                isSorted = false;  // The array is not yet sorted
            }
        }
    }
}

// Function to perform odd-even bubble sort on a GPU
void parallel_bubble_sort_odd_even(vector<int>& arr) {
    int size = arr.size();
    int* d_arr;

    // Allocate memory on the GPU
    cudaMalloc(&d_arr, size * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_arr, arr.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // Set block and grid dimensions for the kernel
    int blockSize = 256;  // Block size
    int gridSize = (size + blockSize - 1) / blockSize;  // Grid size

    // Launch the kernel
    kernel_bubble_sort_odd_even<<<gridSize, blockSize>>>(d_arr, size);

    // Wait for the kernel to complete
    cudaDeviceSynchronize();

    // Copy sorted array back to host
    cudaMemcpy(arr.data(), d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the allocated GPU memory
    cudaFree(d_arr);
}

// Main function to measure performance of sequential and parallel sorts
int main() {
    // Ask for the number of elements
    int n;
    cout << "Enter the number of elements to be sorted: ";
    cin >> n;

    // Random number generator to create random array elements
    random_device rd;  // Obtain a random seed
    mt19937 gen(rd());  // Standard random generator
    uniform_int_distribution<> dis(1, 100);  // Distribution of random numbers from 1 to 100

    // Generate a vector with random elements
    vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = dis(gen);  // Generate a random element
    }

    cout << "Generated array: ";
    for (int i : arr) {
        cout << i << " ";  // Display the unsorted elements
    }
    cout << endl;

    // Measure time for sequential bubble sort
    auto start_seq = chrono::high_resolution_clock::now();
    sequential_bubble_sort(arr);  // Sort using the sequential method
    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed_seq = end_seq - start_seq;

    // Measure time for parallel bubble sort
    vector<int> arr_parallel = arr;  // Make a copy of the original array
    auto start_par = chrono::high_resolution_clock::now();
    parallel_bubble_sort_odd_even(arr_parallel);  // Sort using the parallel method
    auto end_par = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed_par = end_par - start_par;

    // Output the sorted array and timing for both methods
    cout << "Sorted array (sequential): ";
    for (int i : arr) {
        cout << i << " ";  // Display the sorted elements for the sequential sort
    }
    cout << endl;

    cout << "Sorted array (parallel): ";
    for (int i : arr_parallel) {
        cout << i << " ";  // Display the sorted elements for the parallel sort
    }
    cout << endl;

    cout << "Sequential bubble sort time: " << elapsed_seq.count() << " ms" << endl;
    cout << "Parallel bubble sort time: " << elapsed_par.count() << " ms" << endl;

    return 0;
}
