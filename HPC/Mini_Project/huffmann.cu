#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

// Struct for Huffman Tree Nodes
struct HuffmanNode {
    char data; // Character data
    int frequency; // Frequency of the character
    HuffmanNode* left; // Left child in the Huffman Tree
    HuffmanNode* right; // Right child in the Huffman Tree

    HuffmanNode(char data, int frequency)
        : data(data), frequency(frequency), left(nullptr), right(nullptr) {}
};

// Comparator for the priority queue used to build the Huffman Tree
struct Compare {
    bool operator()(HuffmanNode* l, HuffmanNode* r) {
        // Return true if left node has higher frequency than right node
        return l->frequency > r->frequency;
    }
}

// CUDA kernel to count character frequencies in the input array
__global__ void count_frequencies(char* d_input, int* d_freq, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread index
    if (idx < size) { // Ensure index is within bounds
        // Atomic operation to increment the frequency of the character at this index
        atomicAdd(&d_freq[d_input[idx]], 1);
    }
}

// Function to build the Huffman Tree on the CPU (host)
HuffmanNode* build_huffman_tree(const std::vector<int>& freq) {
    // Priority queue to store Huffman Tree nodes
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, Compare> pq;

    // Add all non-zero frequency nodes to the priority queue
    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            pq.push(new HuffmanNode(i, freq[i]));
        }
    }

    // Build the Huffman Tree by repeatedly merging the two smallest nodes
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); // Get the smallest node
        pq.pop(); // Remove it from the queue
        HuffmanNode* right = pq.top(); // Get the second smallest node
        pq.pop(); // Remove it from the queue

        // Create a new parent node with combined frequency
        HuffmanNode* top = new HuffmanNode('\0', left->frequency + right->frequency);
        top->left = left; // Left child is the first smallest node
        top->right = right; // Right child is the second smallest node
        pq.push(top); // Add the new parent node to the queue
    }

    // Return the root of the Huffman Tree
    return pq.top();
}

// Recursive function to generate Huffman Codes from the Huffman Tree
void generate_codes(HuffmanNode* root, std::string code,
                    std::unordered_map<char, std::string>& huffman_codes) {
    if (!root) return; // Base case: if root is null, do nothing

    if (root->data != '\0') { // If it's a leaf node, store the code
        huffman_codes[root->data] = code;
    }

    // Recur for the left child with code appended by '0'
    generate_codes(root->left, code + "0", huffman_codes);

    // Recur for the right child with code appended by '1'
    generate_codes(root->right, code + "1", huffman_codes);
}

// Function to encode the input data using Huffman Codes
std::string encode_data(const std::vector<char>& input,
                        const std::unordered_map<char, std::string>& huffman_codes) {
    std::string encoded;
    // Loop through each character in the input
    for (char c : input) {
        // Add the corresponding Huffman code to the encoded string
        encoded += huffman_codes.at(c);
    }
    return encoded;
}

// Main function to demonstrate Huffman Encoding
int main() {
    std::string input_str = "hello huffman encoding with cuda"; // Input data
    std::vector<char> input(input_str.begin(), input_str.end()); // Convert to vector of chars

    int size = input.size(); // Get the size of the input data

    // Device memory allocation for input array
    char* d_input;
    cudaMalloc(&d_input, size * sizeof(char)); // Allocate memory on GPU
    cudaMemcpy(d_input, input.data(), size * sizeof(char), cudaMemcpyHostToDevice); // Copy data to GPU

    // Device memory allocation for frequency array
    int* d_freq;
    cudaMalloc(&d_freq, 256 * sizeof(int)); // Allocate memory on GPU for frequency array
    cudaMemset(d_freq, 0, 256 * sizeof(int)); // Initialize frequency array to zero

    // Calculate character frequencies using CUDA
    int blockSize = 256; // Number of threads per block
    int gridSize = (size + blockSize - 1) / blockSize; // Number of blocks
    count_frequencies<<<gridSize, blockSize>>>(d_input, d_freq, size); // Launch CUDA kernel

    // Copy frequencies from GPU to CPU (host)
    std::vector<int> h_freq(256, 0); // Create a frequency vector on the host
    cudaMemcpy(h_freq.data(), d_freq, 256 * sizeof(int), cudaMemcpyDeviceToHost); // Copy data back to host

    // Build the Huffman Tree on the CPU (host)
    HuffmanNode* root = build_huffman_tree(h_freq); // Build the Huffman Tree from frequencies

    // Generate Huffman Codes from the Huffman Tree
    std::unordered_map<char, std::string> huffman_codes; // Store Huffman Codes
    generate_codes(root, "", huffman_codes); // Generate codes recursively from the tree

    // Encode the input data using the generated Huffman Codes
    std::string encoded_data = encode_data(input, huffman_codes); // Get the encoded data

    // Output the encoded data
    std::cout << "Encoded data: " << encoded_data << std::endl;

    // Free device memory
    cudaFree(d_input); // Free memory for the input array
    cudaFree(d_freq); // Free memory for the frequency array

    return 0; // Exit the program
}
