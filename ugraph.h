#include <stdio.h>
#include <stdlib.h>

//138paulmiller - Undirected UGraph Adjacency matrix
//All functions a prefixed as denoting g_ graph operation
#define Edge_Type int				
#define Vertex_Type int				

typedef struct _Vertex Vertex;
typedef struct _Edge Edge;
typedef struct _UGraph UGraph;
typedef unsigned int uint;

typedef struct _Vertex{
	Vertex_Type value;
	Vertex* next;
	Edge * edges;
	uint degree;
}Vertex;

typedef struct _Edge{
	Edge_Type value;
	Vertex * vertex;
	Edge* next;
}Edge;


typedef struct _UGraph{
	Vertex * head;
	uint vertex_count;
	uint edge_count;
}UGraph;


void g_new(UGraph * graph);
void g_del(UGraph * graph);
void g_copy(UGraph * src, UGraph * dest);
void g_print(UGraph * graph);

void g_add_vertex(UGraph * graph, Vertex_Type a);
void g_del_vertex(UGraph * graph, Vertex_Type a);
int g_has_vertex(UGraph * graph, Vertex_Type a);

void g_add_edge(UGraph * graph, Vertex_Type a, Vertex_Type b, Edge_Type e);
void g_del_edge(UGraph * graph, Vertex_Type a, Vertex_Type b);
int g_has_edge(UGraph * graph, Vertex_Type a, Vertex_Type b);

//helpers

Vertex * g_find_vertex(UGraph * graph, Vertex_Type a);

Edge * g_find_edge(UGraph * graph, Vertex_Type a, Vertex_Type b);


void g_new(UGraph * graph)
{
	graph->vertex_count = 0;
	graph->edge_count = 0;
	graph->head = 0;
}

void g_del(UGraph * graph)
{
	graph->vertex_count = 0;
	graph->edge_count = 0;
	Edge * edge = 0;
	Edge * temp_e;
	Vertex * vertex = graph->head;
	Vertex *  temp_v;
	while(vertex)
	{
		temp_v = vertex->next;
		edge = vertex->edges;
		while(edge)
		{
			temp_e = edge->next;
			free(edge);
			edge=  temp_e;
		}
		free(vertex);
		vertex = temp_v;
	}
}

void g_print(UGraph * graph)
{
	Edge * edge = 0;
	Vertex * vertex = graph->head;
	while(vertex)
	{
		printf("\n%d: ", vertex->value);
		edge = vertex->edges;
		while(edge)
		{
			printf("[:%d=%d]", edge->vertex->value, edge->value);
			//printf("%d ", edge->vertex->value);
			edge = edge->next;
		}
		vertex = vertex->next;
	}
	puts("");
}


void g_copy(UGraph * src, UGraph * dest)
{
	//TODO
}

void g_add_vertex(UGraph * graph, Vertex_Type a)
{
	//does not check if exists
	
	Vertex * vertex = (Vertex*)malloc(sizeof(Vertex));
	vertex->value = a;
	vertex->next = graph->head;
	vertex->edges = 0;
	graph->head = vertex;
	
}

void g_del_vertex(UGraph * graph, Vertex_Type a)
{
	Vertex * vertex;
	Vertex * trail_vertex;//cur and trailer
	vertex = graph->head;
	while(vertex && vertex->value != a )
	{
		trail_vertex=  vertex;
		vertex = vertex->next;
	}
	if(vertex)
	{
		if(trail_vertex)
		{
			trail_vertex->next = vertex->next;
			vertex->next = 0;
		}
		else//update head
		{
			graph->head = vertex->next;
		}
		// for each edge, go to edges vertex and remove all edges with found vertex
		while(vertex->edges)
		{
		
			Edge * edge = vertex->edges;
			Edge * other_edge=0;
			Edge * trail_edge=0;
			//del all edges
			vertex->edges = edge->next;
			//get edges vertex (other), then remove all instances of a   
			Vertex * other_vertex = edge->vertex;
			other_edge = other_vertex->edges;
			while(other_edge && other_edge->vertex->value != a)
			{
				trail_edge = other_edge;
				other_edge = other_edge->next;
			}
			if(other_edge)
			{
				//must be head or null
				if(trail_edge)
					trail_edge->next = other_edge->next;
				else
					other_vertex->edges = other_edge->next;
				free(other_edge);
			}
			free(edge);
		}
		
		free(vertex);
	}
}
int g_has_vertex(UGraph * graph, Vertex_Type a)
{
	//if not null
	return g_find_vertex(graph, a) != 0; 
}


void g_add_edge(UGraph * graph, Vertex_Type a, Vertex_Type b, Edge_Type e)
{

	//add to both (bidirectional!
	Vertex * vertex_a = g_find_vertex(graph, a);
	Vertex * vertex_b = g_find_vertex(graph, b);
	//Insert if not found?????
	if(vertex_a == 0)
	{
		vertex_a = (Vertex*)malloc(sizeof(Vertex));
		vertex_a->value = a;
		vertex_a->next = graph->head;
		vertex_a->edges = 0;
		graph->head = vertex_a;
	}
	if(vertex_b == 0)
	{
		if(a == b)//do not create new vertex
			vertex_b = vertex_a;
		else
		{
			vertex_b = (Vertex*)malloc(sizeof(Vertex));
			vertex_b->value = b;
			vertex_b->next = graph->head;
			vertex_b->edges = 0;
			graph->head = vertex_b;
		}
	}
	//append to edges for bothe vertices
	//create both edges
	Edge * edge_a = (Edge*)malloc(sizeof(Edge));
	edge_a->value = e;
	edge_a->vertex = vertex_b;
	//append a
	edge_a->next = vertex_a->edges;
	vertex_a->edges = edge_a;
	edge_a = vertex_a->edges;
	if(a != b)
	{
		Edge * edge_b = (Edge*)malloc(sizeof(Edge));
		edge_b->value = e;
		edge_b->vertex = vertex_a;
		//append b
		edge_b->next = vertex_b->edges;
		vertex_b->edges = edge_b;
		edge_b = vertex_b->edges;
	}
}

void g_del_edge(UGraph * graph, Vertex_Type a, Vertex_Type b)
{
	Vertex * vertex_a = g_find_vertex(graph, a);
	Vertex * vertex_b = g_find_vertex(graph, b);
	//if they exist remove each others edges
	if(vertex_a && vertex_b)
	{
		Edge * edge = vertex_a->edges;
		Edge * trail_edge = 0;
		while(edge && edge->vertex->value != b)
		{
			trail_edge = edge;
			edge = edge->next;
		}
		if(edge)
		{
			if(trail_edge)
				trail_edge->next = edge->next;
			else
				vertex_a->edges = edge->next;
			edge->next = 0;
			free(edge);		
			if(vertex_a != vertex_b)
			{
				edge = vertex_b->edges;
				trail_edge = 0;
				while(edge && edge->vertex->value != a)
				{
					trail_edge = edge;
					edge = edge->next;
				}
				if(edge)
				{
					if(trail_edge)
						trail_edge->next = edge->next;
					else
						vertex_b->edges = edge->next;
					edge->next = 0;
					free(edge);
				}
		
			}
			else
			{
				//only remove one instance
			}
		}
		
	
	} 
}


int g_has_edge(UGraph * graph, Vertex_Type a, Vertex_Type b)
{
	//if not null
	return g_find_edge(graph, a, b) != 0; 
}
//------------------- Helpers --------------------------
Vertex * g_find_vertex(UGraph * graph, Vertex_Type a)
{
	Vertex * vertex = graph->head;
	while(vertex && vertex->value != a)
		vertex = vertex->next;
	return vertex;
}

Edge * g_find_edge(UGraph * graph, Vertex_Type a, Vertex_Type b)
{
	Edge * edge = 0;
	Vertex * vertex = graph->head;
	while(vertex && vertex->value != a)
		vertex = vertex->next;
	if(vertex)
	{
		edge = vertex->edges;			
		while(edge && edge->vertex && edge->vertex->value != b){
			edge = edge->next;
		}
	}
	
	return edge;
}
















/*

typedef struct _UGraph {				
	Edge_Type** matrix;	
	
	unsigned int n;//dimensions = max vertex count
	unsigned int vert_count;//vertex count			
	unsigned int edge_count;//edge count				
}UGraph;				

//creates a V x V matrix
void g_new(UGraph *UGraph, unsigned int v);  
//creates a V x V matrix
void g_del(UGraph *g);  
void g_copy(UGraph * dest, UGraph * src);
//print all vertices and edges
void g_print(UGraph *UGraph);

//check if edge exists
int g_is_edge(UGraph *UGraph, unsigned int src_index, unsigned int dest_index);  //modifier,accessor 

Edge_Type g_get_edge(UGraph *UGraph, unsigned int src_index, unsigned int dest_index);  //modifier,accessor 
// will create edge if doesnt exist
void g_set_edge(UGraph *UGraph, unsigned int src_index, unsigned int dest_index, Edge_Type edge);  //modifier,accessor 

//helpers 
void g_grow(UGraph *UGraph);


//---------------------------------Defs---------------------------------------
//creates a V x V matrix
void g_new(UGraph *UGraph, unsigned int n){
	UGraph->n = n;
	UGraph->vert_count = 0;
	UGraph->edge_count = 0;
	UGraph->matrix = (Edge_Type**)malloc(sizeof(Edge_Type*)*n);
	unsigned int j; 
	for(j =0; j < n; ++j){
		UGraph->matrix[j] = (Edge_Type*)malloc(sizeof(Edge_Type)*n);
	} 
} 
//creates a V x V matrix
void g_del(UGraph *UGraph){
	if(UGraph->matrix){
		unsigned int j; 
		for(j =0; j < UGraph->n; ++j){
			free(UGraph->matrix[j]);
		} 
		free(UGraph->matrix);
		UGraph->matrix = 0;
		UGraph->vert_count = UGraph->edge_count = UGraph->n = 0;
	} 
} 

//creates a V x V matrix
void g_print(UGraph *UGraph){
	unsigned int i,j;
	printf("\n[ ");	
	for(i =0; i < UGraph->vert_count; ++i){
		printf("\n [");	
		for(j =0; j < UGraph->vert_count; ++j){
			printf("%3d ", UGraph->matrix[i][j]);
		}
		printf("]\n");	
	} 
	printf("]\n");	
} 

void g_grow(UGraph *UGraph){
	//copy
	UGraph *UGraph_copy=(UGraph*)malloc(sizeof(UGraph));
	unsigned int n = UGraph->n*2;
	unsigned int v = UGraph->vert_count;
	unsigned int e = UGraph->edge_count;
	Edge_Type** matrix = (Edge_Type**)malloc(sizeof(Edge_Type*)*n);
	unsigned int i,j; 
	for(i =0; i < n; ++i){
		matrix[i] = (Edge_Type*)malloc(sizeof(Edge_Type)*n);	

		for(j =0; j < v; ++j){
			if(i < v)
				matrix[i][j] = UGraph->matrix[i][j];
			else
				matrix[i][j] = 0;
		}
	}
	g_del(UGraph);
	UGraph->n = n;
	UGraph->vert_count = v+1;
	UGraph->edge_count = e;
	UGraph->matrix =matrix;
}

int g_is_edge(UGraph *UGraph, unsigned int src_index, unsigned int dest_index){
	return (src_index >= UGraph->vert_count || dest_index >= UGraph->vert_count)
} 
//Returns a reference to the edge object at UGraph[src][dest]
Edge_Type g_get_edge(UGraph* UGraph, unsigned int src_index, unsigned int dest_index){
	//find vertices to get index
	if(! g_is_edge(src_index, dest_index))
		return 0;	
	return UGraph->matrix[src_index][dest_index];
}

void g_set_edge(UGraph* UGraph, unsigned int src_index, unsigned int dest_index, Edge_Type edge){
	//find vertices to get index
	if(! g_is_edge(src_index, dest_index)) //then grow
		g_grow(UGraph);	
	UGraph->matrix[src_index][dest_index] = edge;
}

*/
