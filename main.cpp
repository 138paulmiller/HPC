#include "helper.hpp"
#include <cstddef>
#include <mpi.h>

int main(int argc, char *argv[]) {

    Graph<int, int> original;                 // The original graph.
    Graph<int, int> result;                   // The result graph.
    int world_rank;                           // The unique ID of the process.
    int world_size;                           // The number of processes.
    size_t density;                           // The graph density.
    std::vector<int> sendcounts;              // Contains the number of vertices for each process to work with.
    std::vector<int> strides;                 // Contains the starting index of the vertex vector for each process.
    std::vector<Vertex<int>> receive_buffer;  // Buffer containing the subset of vertices from the graph to work with.
    const int N_VERT = 15;                     // The number of vertices to create.

    // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Initialize graph here!
    if (world_rank == 0) {
        for (int i = 1; i <= N_VERT; i++)
            original.addVertex(i);
    }

    receive_buffer.resize(world_size);
    sendcounts.resize(world_size);
    strides.resize(world_size);

    // Create a new MPI data type for our Vertex struct.
    const int nitems = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Datatype mpi_vertex_type;
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Vertex<int>, value);
    offsets[1] = offsetof(Vertex<int>, index);
    offsets[2] = offsetof(Vertex<int>, degree);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_vertex_type);
    MPI_Type_commit(&mpi_vertex_type);

    // Compute initial density of the graph
    density = original.density();

    // Compute the send counts and stride values.
    computeSendAndStrideCounts(N_VERT, world_size, sendcounts, strides);

    // Print send counts and stride values. *** DEBUG ***
    if (world_rank == 0){
        for (int i = 0; i < world_size; i++) {
            printf("sendcounts[%d] = %d\tdispls[%d] = %d\n", i, sendcounts[i], i, strides[i]);
        }
        std::cout << "\n\n";
    }

    // Divide the data among the processes as described by sendcounts and strides.
    if (sendcounts[world_rank] != 0) {
        MPI_Scatterv(original.toArray(), &sendcounts[0], &strides[0], mpi_vertex_type,
                     &receive_buffer[0], receive_buffer.size(), mpi_vertex_type,
                     0, MPI_COMM_WORLD);
    }
    // Processes that have no work to do will wait until the scatter completes.
    MPI_Barrier(MPI_COMM_WORLD);


    // Print what each process received. *** DEBUG ***
    for (int i = 0; i < sendcounts[world_rank]; i++) {
        printf("P%d: %d\n", world_rank, receive_buffer[i].value);
    }


    MPI_Finalize();
}