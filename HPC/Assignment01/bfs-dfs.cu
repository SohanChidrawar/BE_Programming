#include <iostream>  // Include for input/output operations
#include <vector>    // Include for vector operations
#include <queue>     // Include for queue operations
#include <cuda_runtime.h>  // CUDA runtime header
#include <stack>     // Include for stack operations in host code

using namespace std;

#define V 6  // Define a constant for the maximum number of vertices

// CUDA kernel for Breadth-First Search (BFS) traversal
__global__ void cudaBFS(int* adj_matrix, int* visited, int start, int num_vertices, int* traversal_results) {
    // Only one thread is used in this kernel (threadIdx.x == 0)
    if (threadIdx.x == 0) {
        int queue[256];  // Static array to act as a queue within the kernel
        int front = 0, rear = 0;  // Queue pointers
        queue[rear++] = start;  // Enqueue the start vertex
        visited[start] = 1;  // Mark the start vertex as visited

        // While there are vertices in the queue
        while (front < rear) {
            int current = queue[front++];  // Dequeue the front vertex
            traversal_results[current] = 1;  // Mark as visited in results

            // Traverse adjacent vertices
            for (int j = 0; j < num_vertices; ++j) {
                if (adj_matrix[current * num_vertices + j] && !visited[j]) {  // If adjacent and not visited
                    visited[j] = 1;  // Mark as visited
                    queue[rear++] = j;  // Enqueue
                }
            }
        }
    }
}

// CUDA kernel for Depth-First Search (DFS) traversal
__global__ void cudaDFS(int* adj_matrix, int* visited, int start, int num_vertices, int* traversal_results) {
    // Only one thread is used in this kernel (threadIdx.x == 0)
    if (threadIdx.x == 0) {
        int stack[256];  // Static array-based stack
        int top = 0;  // Stack pointer
        stack[top++] = start;  // Push the start vertex onto the stack
        visited[start] = 1;  // Mark as visited

        // While the stack is not empty
        while (top > 0) {
            int current = stack[--top];  // Pop from the stack
            traversal_results[current] = 1;  // Mark as visited in results

            // Traverse adjacent vertices
            for (int j = 0; j < num_vertices; ++j) {
                if (adj_matrix[current * num_vertices + j] && !visited[j]) {  // If adjacent and not visited
                    visited[j] = 1;  // Mark as visited
                    stack[top++] = j;  // Push onto the stack
                }
            }
        }
    }
}

int main() {
    int num_vertices;
    cout << "Enter the number of vertices: ";
    cin >> num_vertices;  // Input number of vertices

    // Check if number of vertices exceeds the defined maximum
    if (num_vertices > V) {
        cerr << "Error: Number of vertices exceeds the maximum limit (" << V << ")." << endl;
        return 1;  // Exit with error code
    }

    // Allocate memory for the adjacency matrix
    int* adj_matrix = new int[num_vertices * num_vertices];
    memset(adj_matrix, 0, num_vertices * num_vertices * sizeof(int));  // Initialize with zeros

    // Input edges to build the adjacency matrix
    cout << "Enter the number of edges: ";
    int num_edges;
    cin >> num_edges;

    cout << "Enter edges (format: source destination):\n";
    for (int i = 0; i < num_edges; ++i) {
        int source, destination;
        cin >> source >> destination;  // Read the source and destination of each edge
        adj_matrix[source * num_vertices + destination] = 1;  // Set the edge in the adjacency matrix
        adj_matrix[destination * num_vertices + source] = 1;  // Since it's an undirected graph
    }

    // Define the start vertex for traversal
    int start_vertex = 0;

    // Initialize visited arrays and traversal results for BFS and DFS
    int* visited = new int[num_vertices];
    int* bfs_traversal_results = new int[num_vertices];
    int* dfs_traversal_results = new int[num_vertices];
    memset(visited, 0, num_vertices * sizeof(int));  // Initialize with zeros
    memset(bfs_traversal_results, -1, num_vertices * sizeof(int));  // Initialize with -1
    memset(dfs_traversal_results, -1, num_vertices * sizeof(int));  // Initialize with -1

    // Allocate memory on the device (GPU)
    int* d_adj_matrix;
    int* d_visited;
    int* d_bfs_traversal_results;
    int* d_dfs_traversal_results;
    cudaMalloc((void**)&d_adj_matrix, num_vertices * num_vertices * sizeof(int));  // Allocate memory for adjacency matrix
    cudaMalloc((void**)&d_visited, num_vertices * sizeof(int));  // Allocate memory for visited array
    cudaMalloc((void**)&d_bfs_traversal_results, num_vertices * sizeof(int));  // Allocate memory for BFS traversal results
    cudaMalloc((void**)&d_dfs_traversal_results, num_vertices * sizeof(int));  // Allocate memory for DFS traversal results

    // Copy adjacency matrix and visited array to device memory
    cudaMemcpy(d_adj_matrix, adj_matrix, num_vertices * num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events to measure execution time
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Measure BFS execution time
    cudaEventRecord(start_event, 0);  // Start timing
    cudaBFS<<<1, 1>>>(d_adj_matrix, d_visited, start_vertex, num_vertices, d_bfs_traversal_results);  // Perform BFS traversal
    cudaEventRecord(stop_event, 0);  // Stop timing
    cudaDeviceSynchronize();  // Synchronize to ensure kernel execution is complete
    float bfs_elapsed_time;
    cudaEventElapsedTime(&bfs_elapsed_time, start_event, stop_event);  // Get elapsed time

    // Print BFS traversal results and execution time
    cout << "BFS traversal: ";
    cudaMemcpy(bfs_traversal_results, d_bfs_traversal_results, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_vertices; ++i) {
        if (bfs_traversal_results[i] != -1) {
            cout << i << " ";  // Print the visited vertices
        }
    }
    cout << endl;
    cout << "BFS execution time: " << bfs_elapsed_time << " ms" << endl;

    // Reset the visited array for DFS traversal
    memset(visited, 0, num_vertices * sizeof(int));  // Reset to zero
    cudaMemcpy(d_visited, visited, num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    // Measure DFS execution time
    cudaEventRecord(start_event, 0);  // Start timing for DFS
    cudaDFS<<<1, 1>>>(d_adj_matrix, d_visited, start_vertex, num_vertices, d_dfs_traversal_results);  // Perform DFS traversal
    cudaEventRecord(stop_event, 0);  // Stop timing
    cudaDeviceSynchronize();  // Ensure kernel execution is complete
    float dfs_elapsed_time;
    cudaEventElapsedTime(&dfs_elapsed_time, start_event, stop_event);  // Get elapsed time

    // Print DFS traversal results and execution time
    cout << "DFS traversal: ";
    cudaMemcpy(dfs_traversal_results, d_dfs_traversal_results, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_vertices; ++i) {
        if (dfs_traversal_results[i] != -1) {
            cout << i << " ";
        }
    }
    cout << endl;
    cout << "DFS execution time: " << dfs_elapsed_time << " ms" << endl;

    // Cleanup
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaFree(d_adj_matrix);
    cudaFree(d_visited);
    cudaFree(d_bfs_traversal_results);
    cudaFree(d_dfs_traversal_results);

    // Clean up host memory
    delete[] adj_matrix;
    delete[] visited;
    delete[] bfs_traversal_results;
    delete[] dfs_traversal_results;

    return 0;
}



//----------------------------------------------------------------------------------------------------------------------------

// Enter the number of vertices: 6
// Enter the number of edges: 4
// Enter edges (format: source destination):
// 1 2
// 1 3
// 2 4
// 3 5
// BFS traversal: 0 1 2 3 4 5 
// BFS execution time: 51.7145 ms
// DFS traversal: 0 1 2 3 4 5 
// DFS execution time: 0.031648 ms
