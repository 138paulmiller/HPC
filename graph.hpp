/*
	Directed graph class
		- adjacency matrix
		- map each key of where vertex is  
*/
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>


template <typename Vertex_Type, typename Edge_Type>
class Graph{
public:
	struct Vertex{
		Vertex_Type value;
		int index; 	//index within adj matrix and vertices set
	};

	Graph();
	~Graph();

	void addVertex(Vertex_Type a); 
	void removeVertex(Vertex_Type a); 

	
	void addEdge(Vertex_Type a, Vertex_Type b, Edge_Type value); 

	Edge_Type at(Vertex_Type a, Vertex_Type b);
	//gt array of adjacent vertices 
	std::vector<Vertex_Type> adjacent(Vertex_Type a);
	
	void print();
	
private:

	int vertexIndex(Vertex_Type a);
	//all vectors
	std::vector<Vertex> m_vertices;
	//Map key to vertex in vertex vector, used to acheive near O(1) vertex lookup
	std::unordered_map<Vertex_Type, int> m_index_map; 
	//edges
	std::vector<std::vector<Edge_Type> > m_adj_matrix;
};

template <typename Vertex_Type, typename Edge_Type>
Graph<Vertex_Type, Edge_Type>::Graph()

{}

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
		m_index_map[a] = i;
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
		m_index_map.erase(a);
		m_vertices.erase(m_vertices.begin()+i); 
		for(typename std::vector<std::vector<Edge_Type> >::iterator it = m_adj_matrix.begin(); 
							it!= m_adj_matrix.end(); it++){
			it->erase(it->begin()+i);
		}
	}
}

template <typename Vertex_Type, typename Edge_Type>
int Graph<Vertex_Type, Edge_Type>::vertexIndex(Vertex_Type a){
	typename std::unordered_map<Vertex_Type, int> ::iterator it = m_index_map.find(a);
	if(it == m_index_map.end())
		return -1;
	return it->second;
}

template <typename Vertex_Type, typename Edge_Type>
void Graph<Vertex_Type, Edge_Type>::addEdge(Vertex_Type a, Vertex_Type b, Edge_Type value){
	int i = vertexIndex(a);
	if(i != -1){ // exists so del
		int j = vertexIndex(b);
		if(j != -1){ // exists so del
			m_adj_matrix[i][j] = value;
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
	
