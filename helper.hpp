#include "graph.hpp"
#include <cassert>
#include <cmath>


/**
 * This function calculates two things:
 *      1). It computes the number of vertices that each process will work with (N).
 *      2). It computes the starting index that each process will receive N vertices from (the stride value).
 *
 * The results of both of these are stored in @param sendcounts and @param strides.
 *
 * @param num_vertices The number of vertices in the graph.
 * @param world_size The number of processes.
 * @param sendcounts A vector containing the number of vertices to delegate to each process.
 * @param strides A vector containing the starting index for each process to start getting vertices from.
 *
 */
void computeSendAndStrideCounts(int num_vertices, int world_size, std::vector<int> &sendcounts, std::vector<int>& strides){

    int sum = 0;
    int remainder = (num_vertices) % world_size;
    for (int i = 0; i < world_size; i++) {
        sendcounts[i] = (num_vertices) / world_size;
        if (remainder > 0) {
            sendcounts[i]++;
            remainder--;
        }
        strides[i] = sum;
        sum += sendcounts[i];
    }
}

