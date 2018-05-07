#include "helper.hpp"

int main(int argc, char *argv[]) {

    Graph<int, int> original;                     // The original graph.
    Graph<int, int> result;                       // The result graph.
    int world_rank;                               // The unique ID of the process.
    int world_size;                               // The number of processes.
    float density;                                // The graph density.
    std::vector<int> sendcounts;                  // Contains the number of vertices for each process to work with.
    std::vector<int> strides;                     // Contains the starting index of the vertex vector for each process.
    std::vector<Vertex<int>> root_receive_buffer; // Buffer containing the subset of vertices from the graph to work with.
    std::vector<Vertex<int>> proc_receive_buffer; // Buffer containing the subset of vertices from the graph to work with.
    int N_VERT = 11;                              // The number of vertices to create.
    const float EPSILON = 0.1;                    // Heuristic value for removing vertices.

    // MPI Initialization.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Graph initialization.
    if (world_rank == 0) {
        LoadGraph(original, N_VERT);
        density = original.density();
    }

    // Initialize buffers.
    sendcounts.resize(world_size);
    strides.resize(world_size);
    root_receive_buffer.resize(N_VERT);

    // Create a new MPI data type for our Vertex struct.
    MPI_Datatype mpi_vertex_type;
    CreateVertexTypeMPI(mpi_vertex_type);

     while (N_VERT > 0){

        ComputeSendAndStrideCounts(sendcounts, strides, world_size, N_VERT);

        proc_receive_buffer.resize(sendcounts[world_rank]);

        if (sendcounts[world_rank] != 0) {
            MPI_Scatterv(original.toArray(), &sendcounts[0], &strides[0], mpi_vertex_type,
                         &proc_receive_buffer[0], proc_receive_buffer.size(), mpi_vertex_type,
                         0, MPI_COMM_WORLD);
        }
        // Processes that have no work to do will wait until the scatter completes.
        MPI_Barrier(MPI_COMM_WORLD);

        // Broadcast the density to all processes.
        MPI_Bcast(&density, 1, MPI_INT, 0, MPI_COMM_WORLD);

        ComputeVerticesToRemove(proc_receive_buffer, EPSILON, density);

         MPI_Gatherv(&proc_receive_buffer[0], proc_receive_buffer.size(), mpi_vertex_type, &root_receive_buffer[0],
                        &sendcounts[0], &strides[0], mpi_vertex_type, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        if (world_rank == 0){
            int i =0;
            for (auto && v : root_receive_buffer){
                std::cout << v.value << " ";
                if(v.value >= 0)
                    original.removeVertex(v.value);
                else
                    ++i;

            }

            root_receive_buffer.clear();
            root_receive_buffer.resize(i);

            if (original.density() > density){
                density = original.density();
                result = original;
            }

            N_VERT = original.size();
        }

        MPI_Bcast(&N_VERT, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0){
        result.print();
        std::cout << "\nDensity: " << result.density();
    }

    MPI_Finalize();
}
