from typing import List, Set, Dict
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ConceptMapEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate_concept_map(self, 
                           G: nx.DiGraph,
                           reference_concepts: Set[str],
                           reference_relationships: Set[tuple]) -> Dict[str, float]:
        """
        Evaluate the quality of a concept map against reference data.
        
        Args:
            G: Generated concept map graph
            reference_concepts: Set of reference concepts
            reference_relationships: Set of reference relationships (concept1, rel_type, concept2)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract concepts and relationships from the graph
        generated_concepts = set(G.nodes())
        generated_relationships = set((u, G[u][v]['relationship'], v) 
                                   for u, v in G.edges())
        
        # Calculate concept coverage
        concept_coverage = len(generated_concepts.intersection(reference_concepts)) / len(reference_concepts)
        
        # Calculate relationship coverage
        relationship_coverage = len(generated_relationships.intersection(reference_relationships)) / len(reference_relationships)
        
        # Calculate graph density
        density = nx.density(G)
        
        # Calculate average path length
        try:
            avg_path_length = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            avg_path_length = float('inf')
        
        # Calculate clustering coefficient
        clustering_coefficient = nx.average_clustering(G)
        
        # Store metrics
        self.metrics = {
            'concept_coverage': concept_coverage,
            'relationship_coverage': relationship_coverage,
            'density': density,
            'avg_path_length': avg_path_length,
            'clustering_coefficient': clustering_coefficient
        }
        
        return self.metrics
    
    def evaluate_concept_importance(self, 
                                  G: nx.DiGraph,
                                  top_n: int = 10) -> List[tuple]:
        """
        Evaluate the importance of concepts in the map using various centrality measures.
        
        Args:
            G: Concept map graph
            top_n: Number of top concepts to return
            
        Returns:
            List of (concept, importance_score) tuples
        """
        # Calculate various centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Try to calculate eigenvector centrality, use PageRank as fallback
        try:
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        except (nx.NetworkXError, nx.AmbiguousSolution):
            # Use PageRank as a fallback for disconnected graphs
            eigenvector_centrality = nx.pagerank(G)
        
        # Combine centrality measures
        concept_importance = {}
        for node in G.nodes():
            importance = (
                degree_centrality[node] +
                betweenness_centrality[node] +
                eigenvector_centrality[node]
            ) / 3
            concept_importance[node] = importance
        
        # Sort concepts by importance
        sorted_concepts = sorted(concept_importance.items(),
                               key=lambda x: x[1],
                               reverse=True)
        
        return sorted_concepts[:top_n]
    
    def evaluate_relationship_types(self, G: nx.DiGraph) -> Dict[str, int]:
        """
        Analyze the distribution of relationship types in the concept map.
        
        Args:
            G: Concept map graph
            
        Returns:
            Dictionary mapping relationship types to their counts
        """
        relationship_counts = {}
        for _, _, data in G.edges(data=True):
            rel_type = data['relationship']
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        return relationship_counts

# Example usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.DiGraph()
    G.add_edges_from([
        ("Machine Learning", "Deep Learning", {"relationship": "is_a"}),
        ("Deep Learning", "Neural Networks", {"relationship": "has_part"}),
        ("Machine Learning", "Supervised Learning", {"relationship": "has_part"}),
        ("Supervised Learning", "Input", {"relationship": "used_for"}),
        ("Supervised Learning", "Output", {"relationship": "used_for"})
    ])
    
    # Create reference data
    reference_concepts = {
        "Machine Learning", "Deep Learning", "Neural Networks",
        "Supervised Learning", "Input", "Output", "Function"
    }
    
    reference_relationships = {
        ("Machine Learning", "is_a", "Deep Learning"),
        ("Deep Learning", "has_part", "Neural Networks"),
        ("Machine Learning", "has_part", "Supervised Learning"),
        ("Supervised Learning", "used_for", "Input"),
        ("Supervised Learning", "used_for", "Output")
    }
    
    # Create evaluator
    evaluator = ConceptMapEvaluator()
    
    # Evaluate concept map
    metrics = evaluator.evaluate_concept_map(G, reference_concepts, reference_relationships)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Evaluate concept importance
    important_concepts = evaluator.evaluate_concept_importance(G)
    print("\nTop Important Concepts:")
    for concept, importance in important_concepts:
        print(f"{concept}: {importance:.4f}")
    
    # Evaluate relationship types
    relationship_distribution = evaluator.evaluate_relationship_types(G)
    print("\nRelationship Type Distribution:")
    for rel_type, count in relationship_distribution.items():
        print(f"{rel_type}: {count}") 