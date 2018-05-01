/*
	Directed graph class
		- adjacency matrix
		- map each key of where vertex is  
*/
#include <vector>
#include <iostream>
#include <iomanip>

template <typename Vertex_Type>
struct Vertex{
	Vertex_Type value;
	int index; 	//index within adj matrix and vertices set
};

template <typename Vertex_Type, typename Edge_Type>
class Graph{
public:

	Graph();
	~Graph();

	void addVertex(Vertex_Type a);
	void removeVertex(Vertex_Type a);
	void addEdge(Vertex_Type a, Vertex_Type b, Edge_Type value);

	Edge_Type at(Vertex_Type a, Vertex_Type b);
	std::vector<Vertex_Type> adjacent(Vertex_Type a); //gt array of adjacent vertices

	void print();
	bool isEmpty() const;
	size_t density() const;
	size_t size() const;

private:

	int vertexIndex(Vertex_Type a);
	std::vector<Vertex> m_vertices; //all vectors
	std::vector<std::vector<Edge_Type>> m_adj_matrix; //edges
	size_t num_edges;
};

template <typename Vertex_Type, typename Edge_Type>
Graph<Vertex_Type, Edge_Type>::Graph()
{
	num_edges = 0;
}

template <typename Vertex_Type, typename Edge_Type>
Graph<Vertex_Type, Edge_Type>::~Graph(){}

template <typename Vertex_Type, typename Edge_Type>
void Graph<Vertex_Type, Edge_Type>::print(){
	std::cout << std::setw(5) << ' '<< '|';
	for(typename std::vector<Vertex>::iterator vertIt  = m_vertices.begin(); vertIt != m_vertices.end(); vertIt++)
		std::cout << std::setw(5) << vertIt->value;
	std::cout << '\n' << std::setw(5) << ' ' << '|' << std::setfill('-');
	for(typename std::vector<Vertex>::iterator vertIt  = m_vertices.begin(); vertIt != m_vertices.end(); vertIt++)
		std::cout << std::setw(5) << '-';
	std::cout << '\n' << std::setfill(' ');
	for(typename std::vector<Vertex>::iterator vertIt  = m_vertices.begin(); vertIt != m_vertices.end(); vertIt++){
		std::vector<Edge_Type> edges = m_adj_matrix[vertIt->index];
		std::cout << std::setw(5) << vertIt->value << "|";
		for(typename std::vector<Edge_Type>::iterator it = edges.begin(); it != edges.end(); it++)
			std::cout << std::setw(5) << *it;
		std::cout << '\n';
	}

}



template <typename Vertex_Type, typename Edge_Type>
void Graph<Vertex_Type, Edge_Type>::addVertex(Vertex_Type a){
	int i = vertexIndex(a);
	if(i == -1){ //does not exist so add
		i = m_vertices.size();
		Vertex v = {a, i};
		m_vertices.push_back(v);
		m_adj_matrix.push_back(std::vector<Edge_Type>());
		for(typename std::vector<std::vector<Edge_Type> >::iterator it = m_adj_matrix.begin();
			it!= m_adj_matrix.end(); it++){
			it->resize(i+1);
		}

	}
}

template <typename Vertex_Type, typename Edge_Type>
void Graph<Vertex_Type, Edge_Type>::removeVertex(Vertex_Type a){
	int i = vertexIndex(a);
	if(i != -1){ // exists so del
		m_vertices.erase(m_vertices.begin()+i);

		for(typename std::vector<std::vector<Edge_Type> >::iterator it = m_adj_matrix.begin();
			it!= m_adj_matrix.end(); it++){
			it->erase(it->begin()+i);
		}
		//update all indices starting at i
		for(typename std::vector<Vertex>::iterator it = m_vertices.begin()+i; it != m_vertices.end(); it++)
			it->index--;
	}
}

template <typename Vertex_Type, typename Edge_Type>
int Graph<Vertex_Type, Edge_Type>::vertexIndex(Vertex_Type a){
	for(typename std::vector<Vertex>::iterator it = m_vertices.begin(); it != m_vertices.end(); it++)
		if(it->value == a)
			return it->index;
	return -1;
}

template <typename Vertex_Type, typename Edge_Type>
void Graph<Vertex_Type, Edge_Type>::addEdge(Vertex_Type a, Vertex_Type b, Edge_Type value){
	int i = vertexIndex(a);
	if(i != -1){ // exists so del
		int j = vertexIndex(b);
		if(j != -1){ // exists so del
			m_adj_matrix[i][j] = value;
			num_edges++;
		}
	}
}
template <typename Vertex_Type, typename Edge_Type>
Edge_Type Graph<Vertex_Type, Edge_Type>::at(Vertex_Type a, Vertex_Type b){
	int i = vertexIndex(a);
	if(i != -1){ // exists so del
		int j = vertexIndex(b);
		if(j != -1){ // exists so del
			return m_adj_matrix[i][j];
		}
	}
	return Edge_Type();
}

//gt array of adjacent vertices
template <typename Vertex_Type, typename Edge_Type>
std::vector<Vertex_Type> Graph<Vertex_Type, Edge_Type>::adjacent(Vertex_Type a){
	std::vector<Vertex_Type> adj;
	int i = vertexIndex(a);
	if(i != -1){ // exists so del
		std::vector<Edge_Type> edges = m_adj_matrix[i];
		for(typename std::vector<Vertex>::iterator vertIt  = m_vertices.begin(); vertIt != m_vertices.end(); vertIt++){
			int j = vertIt->index;
			if( m_adj_matrix[i][j] != Edge_Type() || m_adj_matrix[j][i] != Edge_Type() )
				adj.push_back(vertIt->value);
		}
	}
	return adj;
}
/**
 * Computes and returns the density of the graph. The density function is
 * simply the number of edges divided by the number of vertices, or D = E/V.
 *
 * @return The density of the graph.
 */
template <typename Vertex_Type, typename Edge_Type>
size_t Graph<Vertex_Type, Edge_Type>::density() const {
	return isEmpty() ? 0 : (num_edges / m_vertices.size());
}

/**
 * Returns true if this graph contains no elements.
 *
 * @return True if the graph is empty, false if not.
 */
template <typename Vertex_Type, typename Edge_Type>
bool Graph<Vertex_Type, Edge_Type>::isEmpty() const {
	return m_vertices.empty();
}

/**
 * Returns the size of the graph (number of vertices).
 *
 * @return The number of vertices.
 */
template <typename Vertex_Type, typename Edge_Type>
size_t Graph<Vertex_Type, Edge_Type>::size() const {
	return m_vertices.size();
}


	
