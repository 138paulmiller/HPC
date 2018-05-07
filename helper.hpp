#include "graph.hpp"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <mpi.h>


/**
 * This function calculates two things:
 *      1). It computes the number of vertices that each process will work with (N).
 *      2). It computes the starting index that each process will receive N vertices from (the stride value).
 *
 * The results of both of these are stored in @param sendcounts and @param strides.
 *
 * @param sendcounts A vector containing the number of vertices to delegate to each process.
 * @param strides A vector containing the starting index for each process to start getting vertices from.
 * @param world_size The number of processes.
 * @param num_vertices The number of vertices in the graph.
 */
void ComputeSendAndStrideCounts(std::vector<int> &sendcounts, std::vector<int>& strides, int world_size, int num_vertices){

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


/**
 * Determines which vertices are eligible for removal. A vertex V is eligible for removal
 * from Graph G if the following inequality holds true: degree(V) <= 2(1 + epsilon) * density(G).
 *
 * @param vertices A vector of vertex objects.
 * @param epsilon The heuristic value (variable e from the algorithm in the book).
 * @return A vector of vertices that are eligible for removal.
 */
template <typename T>
void ComputeVerticesToRemove(std::vector<Vertex<T>>& vertices, int epsilon, double density){
    for (auto & v : vertices){
        //if keep set to -1
        if (v.degree > 2 * (1 + epsilon) * density){
            v.value = -1;
        }
    }
}

/**
 * Helper function to construct a custom MPI object.
 *
 * @param mpi_vertex_type an MPI_Datatype reference.
 */
void CreateVertexTypeMPI(MPI_Datatype& mpi_vertex_type){
    const int nitems = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Vertex<int>, value);
    offsets[1] = offsetof(Vertex<int>, index);
    offsets[2] = offsetof(Vertex<int>, degree);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_vertex_type);
    MPI_Type_commit(&mpi_vertex_type);
}


#define EDGE(a,b) g.addEdge(a, b, 1);

/**
 * TODO: Rewrite this function or come up with a more dynamic way of initializing the graph.
 *
 * @param g An empty graph to fill.
 * @param N The number of vertices to fill the graph with.
 */
template <typename V, typename E>
void LoadGraph(Graph<V,E>& g, int N){
#include "graph"
};


void debug(int rank, const std::string &msg){
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << msg;
    MPI_Barrier(MPI_COMM_WORLD);
}