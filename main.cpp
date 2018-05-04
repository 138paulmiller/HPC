#include "helper.hpp"

int main(int argc, char *argv[]) {

    Graph<int, int> original;                 // The original graph.
    Graph<int, int> result;                   // The result graph.
    int world_rank;                           // The unique ID of the process.
    int world_size;                           // The number of processes.
    float density;                            // The graph density.
    std::vector<int> sendcounts;              // Contains the number of vertices for each process to work with.
    std::vector<int> strides;                 // Contains the starting index of the vertex vector for each process.
    std::vector<Vertex<int>> receive_buffer;  // Buffer containing the subset of vertices from the graph to work with.
    const int N_VERT = 11;                    // The number of vertices to create.
    const int EPSILON = 1;                    // Heuristic value for removing vertices.
    int LOOP = 1;                             // LCV; set to 0 to exit while loop.

    // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Graph initialization (figure 11.1 from book).
    if (world_rank == 0) {
        LoadGraph(original, N_VERT);
        density = original.density();
    }

    // Initialize buffers.
    receive_buffer.resize(world_size);
    sendcounts.resize(world_size);
    strides.resize(world_size);

    // Create a new MPI data type for our Vertex struct.
    MPI_Datatype mpi_vertex_type;
    CreateVertexTypeMPI(mpi_vertex_type);

    // Will continue looping until the original graph is empty.
    while (LOOP){

        ComputeSendAndStrideCounts(sendcounts, strides, world_size, original.size());

        // *** DEBUG ***
        if (world_rank == 0){
            for (int i = 0; i < world_size; i++) { printf("sendcounts[%d] = %d\tdispls[%d] = %d\n", i, sendcounts[i], i, strides[i]); }
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

        // Broadcast the density to all processes.
        MPI_Bcast(&density, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // *** DEBUG ***
        for (int i = 0; i < sendcounts[world_rank]; i++) { printf("P%d: %d\n", world_rank, receive_buffer[i].value); }

        auto verticesToRemove = ComputeVerticesToRemove(receive_buffer, EPSILON, density);

        // *** DEBUG ***
        for (int i = 0; i < verticesToRemove.size(); i++) { printf("\nP%d: %d", world_rank, verticesToRemove[i].value); }


        // We are going to reuse these buffers, so make sure they are clear. (might just be able to directly write over these?)
        sendcounts.clear();
        strides.clear();

        // Recompute the send and stride counts for when we gather all of the vertices to remove to the root process.
        ComputeSendAndStrideCounts(sendcounts, strides, world_size, verticesToRemove.size());

        MPI_Gatherv(&verticesToRemove[0], verticesToRemove.size(), mpi_vertex_type, &receive_buffer[0],
                    &sendcounts[0], &strides[0], mpi_vertex_type, 0, MPI_COMM_WORLD);

        // Remove vertices from graph update the graph density.
        if (world_rank == 0){
            for (auto & v : receive_buffer){
                original.removeVertex(v.value);
            }
            if (original.density() > density){
                density = original.density();
                result = original;
            }
            // Update LCV in all processes.
            if (original.isEmpty()){
                LOOP = 0;
                MPI_Bcast(&LOOP, 1, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Print results.
    if (world_rank == 0){
        result.print();
        std::cout << "\nDensity: " << result.density();
    }

    MPI_Finalize();
}