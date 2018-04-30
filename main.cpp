#include "graph.hpp"


int main(){

	Graph<int, int> g;
	for(int i = 0; i < 10; i++){
		g.addVertex(i );
		for(int j = 0; j < 10; j++){
			g.addEdge(i,j, i*j+1);
			g.addEdge(j,i, i*j);
		}
	}		
	g.print();
	std::cout << '\n';
	g.removeVertex(3);
	g.print();
	std::cout << '\n';
	std::vector<int> adj = g.adjacent(2);
	for(typename std::vector<int>::iterator adjIt = adj.begin(); adjIt != adj.end(); adjIt++)
		std::cout << " " << *adjIt;
}
