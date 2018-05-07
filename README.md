Hassan Hamod, Paul Miller, Blake Molina 
CPSC 479 
PROJECT 02 - FINDING DENSEST SUBGRAPH 
Dr. Doina Bein

---------- TO RUN THIS PROJECT ------

docker run --rm -it -v $(pwd):/project nlknguyen/alpine-mpich

mpic++ -std=c++11 main.cpp

mpirun -n 5 ./a.out
