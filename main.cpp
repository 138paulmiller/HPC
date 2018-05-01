#include "graph.hpp"
#include <mpi.h>

/**
 * Returns a unique subset of vertices from a non-empty graph object, based on the world_size
 * and world_rank parameters.
 *
 * @param graph A non-empty graph object.
 * @param rank The world rank of process.
 * @return A unique subset of vertices from the graph.
 */
std::vector<Vertex> PartitionGraph(Graph<int,int> &graph, int world_size, int world_rank){
    assert(!graph.isEmpty());
    assert(world_rank >= 0);

    size_t num_vertices = graph.size() / world_size;
    std::vector<Vertex> result;
    result.reserve(num_vertices);

    // Get vertices from the graph. Offsetting by the world_rank retrieves distinct vertices.
    for (int i = 0; i < num_vertices; i++){
        // TODO: Get vertices from the graph
    }

    return result;
};

int main(int argc, char** argv){

    Graph<int, int> original;
    Graph<int, int> result;
    int world_rank;
    int world_size;
    size_t density;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Make sure there aren't more processes than vertices in the graph.
    assert(original.size() > world_size);

    density = original.density();

    if (world_rank != 0){
        auto sub_graph = PartitionGraph(original, world_size, world_rank);
        for (auto const& vertex : sub_graph){
            // TODO: Compute degree of each vertex and if it's <= 2(1+e) * p(S), return the vertices to remove to process 0.
        }

         // TODO: Send information back to process 0 so that it can reduce the map

    } else{
        // TODO: Collect information from child processes
        // TODO: Reduce graph, calculate density (P) and store new graph in 'result' if P > the before P value.
        // TODO: Continue looping until the original graph is empty.
    }

    result.print();

    MPI_Finalize();
}
