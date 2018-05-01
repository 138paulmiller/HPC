#include "graph.hpp"

//c++11
int main(){

	Graph<int, int> g;
	for(int i = 0; i < 10; i++){
		g.addVertex(i );
		for(int j = 0; j < 10; j++){
			g.addEdge(i,j, i);
			g.addEdge(j,i, j);
		}
	}		
	g.print();
	std::cout << '\n';
	g.removeVertex(3);
	g.removeVertex(0);
	g.removeVertex(9);
	g.removeVertex(7);
	g.addVertex(3);
	g.addEdge(3,5, 10);
	g.print();
	std::cout << '\n';
	std::vector<int> adj = g.adjacent(2);
	for(typename std::vector<int>::iterator adjIt = adj.begin(); adjIt != adj.end(); adjIt++)
		std::cout << " " << *adjIt;
}
