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
	size_t index; 	//index within adj matrix and vertices set
};

template <typename Vertex_Type, typename Edge_Type>
class Graph{
public:

	Graph();
	~Graph();

	void addVertex(Vertex_Type a);
	void removeVertex(Vertex_Type a);
	void addEdge(Vertex_Type a, Vertex_Type b, Edge_Type value);
	void print() const;

	Edge_Type at(Vertex_Type a, Vertex_Type b);
	std::vector<Vertex_Type> adjacent(Vertex_Type a); //gt array of adjacent vertices
    Vertex<Vertex_Type>* get(Vertex_Type a) const;

	int indexOf(Vertex_Type a) const;

	bool isEmpty() const;
	bool contains(Vertex_Type key) const;

	size_t density() const;
	size_t size() const;

private:

	std::vector<Vertex<Vertex_Type>> m_vertices; //all vectors
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
void Graph<Vertex_Type, Edge_Type>::print() const{
	std::cout << std::setw(5) << ' '<< '|';
	for(auto const& v : m_vertices)
		std::cout << std::setw(5) << v.value;
	std::cout << '\n' << std::setw(5) << ' ' << '|' << std::setfill('-');
	for(auto const& v : m_vertices)
		std::cout << std::setw(5) << '-';
	std::cout << '\n' << std::setfill(' ');
	for(auto const& v : m_vertices){
		std::vector<Edge_Type> edges = m_adj_matrix[v.index];
		std::cout << std::setw(5) << v.value << "|";
		for(auto const& e : edges)
			std::cout << std::setw(5) << e;
		std::cout << '\n';
	}

}


/**
 * Adds a vertex with value of 'a' to the graph. Note that 'a' must be unique,
 * otherwise the vertex will not be added.
 *
 * @param a The unique value of the vertex.
 */
template <typename Vertex_Type, typename Edge_Type>
void Graph<Vertex_Type, Edge_Type>::addVertex(Vertex_Type a){
	if(!contains(a)){ // does not exist so add
		size_t i = m_vertices.size();
		Vertex<Vertex_Type> v = {a, i};
		m_vertices.push_back(v);
		m_adj_matrix.push_back(std::vector<Edge_Type>());
		for(typename std::vector<std::vector<Edge_Type> >::iterator it = m_adj_matrix.begin();
			it!= m_adj_matrix.end(); it++){
			it->resize(i+1);
		}

	}
}

/**
 * Removes the vertex with value a from the graph.
 *
 * @param a The unique value of the vertex.
 */
template <typename Vertex_Type, typename Edge_Type>
void Graph<Vertex_Type, Edge_Type>::removeVertex(Vertex_Type a){
	int i = indexOf(a);
	if(i != -1){ // exists so del

		m_vertices.erase(m_vertices.begin() + i);

		for(typename std::vector<std::vector<Edge_Type> >::iterator it = m_adj_matrix.begin();
			it!= m_adj_matrix.end(); it++){
			it->erase(it->begin() + i);
		}

		//update all indices starting at i
		for(typename std::vector<Vertex<Vertex_Type>>::iterator it = m_vertices.begin() + i; it != m_vertices.end(); it++)
			it->index--;
	}
}

/**
 * Adds an edge between two vertices.
 *
 * @param a The unique value of the first vertex.
 * @param b The unique value of the second vertex.
 * @param value The weight of the edge.
 */
template <typename Vertex_Type, typename Edge_Type>
void Graph<Vertex_Type, Edge_Type>::addEdge(Vertex_Type a, Vertex_Type b, Edge_Type value){
	int i = indexOf(a);
	int j = indexOf(b);
	if(i != -1 && j != -1){
		m_adj_matrix[i][j] = value;
		m_adj_matrix[j][i] = value;
		num_edges++;
	}
}

/**
 * TODO:
 */
template <typename Vertex_Type, typename Edge_Type>
Edge_Type Graph<Vertex_Type, Edge_Type>::at(Vertex_Type a, Vertex_Type b){
	int i = indexOf(a);
	int j = indexOf(b);
	if(i != -1 && j != -1){ // exists so del
		return m_adj_matrix[i][j];
	}
	return Edge_Type();
}

/**
 * Returns a vector of adjacent vertices to the vertex with value a.
 *
 * @param a The unique value of the vertex.
 * @return A vector of adjacent vertices. May be empty.
 */
template <typename Vertex_Type, typename Edge_Type>
std::vector<Vertex_Type> Graph<Vertex_Type, Edge_Type>::adjacent(Vertex_Type a){
	std::vector<Vertex_Type> adj;
	int i = indexOf(a);

	if(i != -1){ // exists so del
		std::vector<Edge_Type> edges = m_adj_matrix[i];
		for(typename std::vector<Vertex<Vertex_Type>>::iterator vertIt  = m_vertices.begin(); vertIt != m_vertices.end(); vertIt++){
			int j = vertIt->index;
			if( m_adj_matrix[i][j] != Edge_Type() || m_adj_matrix[j][i] != Edge_Type() )
				adj.push_back(vertIt->value);
		}
	}
	return adj;
}

/**
 * Returns the vertex with the value a, if found.
 *
 * @param a The unique value of the vertex.
 * @return A pointer to the vertex if found, else null.
 */
template <typename Vertex_Type, typename Edge_Type>
Vertex<Vertex_Type>* Graph<Vertex_Type, Edge_Type>::get(Vertex_Type a) const{
	Vertex<Vertex_Type>* result = nullptr;
    for(auto v : m_vertices){
		if (v.value == a){
			result = &v;
			break;
		}
	}
	return result;
}

/**
 * Returns the index of the first occurrence of the specified vertex in this graph,
 * or -1 if this graph does not contain the vertex.
 *
 * @param a The unique value of the vertex.
 * @return The index of the vertex if found, else -1.
 */
template <typename Vertex_Type, typename Edge_Type>
int Graph<Vertex_Type, Edge_Type>::indexOf(Vertex_Type a) const{
	auto v = get(a);
	return v != nullptr ? v->index : -1;
}

/**
 * Returns true if this graph contains the specified vertex.
 *
 * @param key The unique value of the vertex.
 * @return True, if found, false if not.
 */
template <typename Vertex_Type, typename Edge_Type>
bool Graph<Vertex_Type, Edge_Type>::contains(Vertex_Type key) const{
	return get(key) != nullptr;
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


	
