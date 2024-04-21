%%writefile depthfirst.cu

#include <iostream>
#include <vector>
#include <stack>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

// CUDA kernel to perform parallel DFS
__global__ void parallel_dfs(int *adj_list, int *adj_list_offsets, int *visited, int start_node, int num_nodes) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < num_nodes) {
        if (thread_id == start_node) {
            visited[start_node] = 1;  // Mark the starting node as visited
            __threadfence();  // Ensure all threads see the update
        }

        __shared__ int stack[THREADS_PER_BLOCK];
        int top = -1;

        // Push the start node onto the stack (only for the initial thread)
        if (thread_id == start_node) {
            stack[++top] = start_node;
        }

        while (top >= 0) {
            int current_node = stack[top--];  // Pop from the stack

            // Traverse neighbors
            for (int i = adj_list_offsets[current_node]; i < adj_list_offsets[current_node + 1]; ++i) {
                int neighbor = adj_list[i];
                if (!visited[neighbor]) {
                    visited[neighbor] = 1;  // Mark as visited
                    stack[++top] = neighbor;  // Push onto the stack
                    __threadfence();  // Ensure other threads see the update
                }
            }
        }
    }
}

int main() {
    int num_nodes = 6;  // Number of nodes in the graph
    std::vector<std::vector<int>> adj_list = {
        {1, 2},   // Node 0 connected to 1, 2
        {0, 3, 4},  // Node 1 connected to 0, 3, 4
        {0, 4, 5},  // Node 2 connected to 0, 4, 5
        {1},       // Node 3 connected to 1
        {1, 2},    // Node 4 connected to 1, 2
        {2}        // Node 5 connected to 2
    };

    int *d_adj_list, *d_adj_list_offsets, *d_visited;
    int start_node = 0;  // Starting node for DFS

    // Create a flat adjacency list and offsets
    std::vector<int> flat_adj_list;
    std::vector<int> adj_list_offsets(num_nodes + 1, 0);

    int offset = 0;
    for (int i = 0; i < num_nodes; i++) {
        adj_list_offsets[i] = offset;
        flat_adj_list.insert(flat_adj_list.end(), adj_list[i].begin(), adj_list[i].end());
        offset += adj_list[i].size();
    }
    adj_list_offsets[num_nodes] = offset;

    // Allocate and copy data to GPU
    cudaMalloc(&d_adj_list, flat_adj_list.size() * sizeof(int));
    cudaMemcpy(d_adj_list, flat_adj_list.data(), flat_adj_list.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_adj_list_offsets, adj_list_offsets.size() * sizeof(int));
    cudaMemcpy(d_adj_list_offsets, adj_list_offsets.data(), adj_list_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_visited, num_nodes * sizeof(int));
    cudaMemset(d_visited, 0, num_nodes * sizeof(int));  // Initialize visited to 0

    // Launch the parallel DFS kernel
    int num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    parallel_dfs<<<num_blocks, THREADS_PER_BLOCK>>>(d_adj_list, d_adj_list_offsets, d_visited, start_node, num_nodes);

    // Retrieve the visited array from the GPU
    std::vector<int> visited(num_nodes, 0);
    cudaMemcpy(visited.data(), d_visited, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

    // Output visited nodes
    std::cout << "Visited nodes in DFS order: ";
    for (int i = 0; i < num_nodes; i++) {
        if (visited[i]) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    // Free GPU memory
    cudaFree(d_adj_list);
    cudaFree(d_adj_list_offsets);
    cudaFree(d_visited);

    return 0;
}
