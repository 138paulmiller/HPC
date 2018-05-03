#include "graph.hpp"
#include <cassert>
#include <cmath>


/**
 * This function calculates two things:
 *      1). It computes the number of vertices that each process will work with (N).
 *      2). It computes the starting index that each process will receive N vertices from (the offset).
 *
 * The results of both of these are stored in @param send_counts and @param offset.
 *
 * @param graph A graph object.
 * @param world_size The number of processes.
 * @param send_counts A vector containing the number of vertices to delegate to each process.
 * @param offset A vector containing the starting index for each process to start getting vertices from.
 *
 */
void computeSendCounts(Graph<int, int> &graph, int world_size, std::vector<int> &send_counts, std::vector<int>& offset){

    // If there are less vertices than processes, process 0 will be delegated with all of the vertices.
    if(graph.size() / world_size < 1){
        send_counts.resize(1, graph.size());
        offset.resize(1,0);
    }
    else {
        int v_per_rank = (int)ceil(graph.size() / world_size);
        send_counts.resize(world_size, v_per_rank);
        offset.resize(world_size);

        send_counts[world_size - 1] = graph.size() - (world_size - 1) * v_per_rank;
        int temp = 0;
        for (int i = 0; i < world_size; ++i) {
            offset[i] = temp;
            temp += send_counts[i];
        }
    }
}

