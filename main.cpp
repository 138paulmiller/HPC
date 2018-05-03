#include "helper.hpp"
#include <cstddef>
#include <mpi.h>



int main(int argc, char** argv) {

    Graph<int, int> original;                   // The original graph.
    Graph<int, int> result;                     // The result graph.
    int world_rank;                             // The unique ID of the process.
    int world_size;                             // The number of processes.
    size_t density;                             // The graph density.
    std::vector<int> send_counts;               // Contains the number of vertices for each process to work with.
    std::vector<int> offsets;                    // Contains the starting index of the vertex vector for each process.
    std::vector<Vertex<int>> receive_buffer;    // Buffer containing the subset of vertices from the graph to work with.
    const int N_VERT = 3;                       // The number of vertices to create.

    // MPI Initialization
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Initialize graph here!
    if (world_rank == 0) {
        for (int i = 1; i <= N_VERT; i++)
            original.addVertex(i);
    }


    // Create a new MPI data type for our Vertex struct.
    const int nitems = 3;
    int blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Datatype mpi_vertex_type;
    MPI_Aint struct_offsets[3];
    struct_offsets[0] = offsetof(Vertex<int>, value);
    struct_offsets[1] = offsetof(Vertex<int>, index);
    struct_offsets[2] = offsetof(Vertex<int>, degree);
    MPI_Type_create_struct(nitems, blocklengths, struct_offsets, types, &mpi_vertex_type);
    MPI_Type_commit(&mpi_vertex_type);

    // Compute initial density of the graph
    density = original.density();

    if (world_rank == 0) {
        computeSendCounts(original, world_size, send_counts, offsets);

        std::cout << "\nSENDS:\n";
        for (int i = 0; i < send_counts.size(); i++) {
            std::cout << send_counts[i] << " ";
        }
        std::cout << "\nDISP:\n";
        for (int i = 0; i < offsets.size(); i++) {
            std::cout << offsets[i] << " ";
        }
        std::cout << "\n";
    }
    receive_buffer.resize(world_size);
    if(send_counts.size() > 1) {

        MPI_Scatterv(original.toArray(), &send_counts[0], &offsets[0], mpi_vertex_type,
                     &receive_buffer[0], receive_buffer.size(), mpi_vertex_type,
                     0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        for (int i = 0; i < receive_buffer.size(); i++){
            std::cout << "\nP" << world_rank << ": " << receive_buffer[i].value;
        }
    }
        // only one, root will handle
    else{
        if(world_rank == 0){
            for (int i = 0; i < original.size(); i++){
                std::cout << "\nP" << world_rank << ": " << original.toArray()[i].value;
            }
        }
    }

    MPI_Finalize();
}