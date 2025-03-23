import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
import re

class RelationshipExtractor:
    def __init__(self):
        self.relationship_patterns = {
            'is_a': [
                r'(\w+)\s+is\s+a\s+(\w+)',
                r'(\w+)\s+is\s+an\s+(\w+)',
                r'(\w+)\s+is\s+the\s+(\w+)'
            ],
            'has_part': [
                r'(\w+)\s+has\s+(\w+)',
                r'(\w+)\s+contains\s+(\w+)',
                r'(\w+)\s+includes\s+(\w+)'
            ],
            'used_for': [
                r'(\w+)\s+is\s+used\s+for\s+(\w+)',
                r'(\w+)\s+can\s+be\s+used\s+to\s+(\w+)'
            ]
        }
        
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english'
        )

    def extract_relationships(self, texts: List[str], concepts: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between concepts using both pattern matching and co-occurrence analysis.
        
        Args:
            texts: List of text segments to analyze
            concepts: List of concepts to find relationships between
            
        Returns:
            List of tuples (concept1, relationship_type, concept2)
        """
        relationships = []
        
        # Pattern-based relationship extraction
        pattern_relationships = self._extract_pattern_relationships(texts, concepts)
        relationships.extend(pattern_relationships)
        
        # Co-occurrence based relationship extraction
        cooccurrence_relationships = self._extract_cooccurrence_relationships(texts, concepts)
        relationships.extend(cooccurrence_relationships)
        
        return relationships

    def _extract_pattern_relationships(self, texts: List[str], concepts: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships using predefined patterns."""
        relationships = []
        concept_set = set(concepts)
        
        for text in texts:
            for rel_type, patterns in self.relationship_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text.lower())
                    for match in matches:
                        concept1, concept2 = match.groups()
                        if concept1 in concept_set and concept2 in concept_set:
                            relationships.append((concept1, rel_type, concept2))
        
        return relationships

    def _extract_cooccurrence_relationships(self, texts: List[str], concepts: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships based on concept co-occurrence in sentences."""
        relationships = []
        concept_set = set(concepts)
        
        # Create document-term matrix
        dtm = self.vectorizer.fit_transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Create concept indices mapping
        concept_indices = {concept: idx for idx, concept in enumerate(feature_names) if concept in concept_set}
        
        # Calculate co-occurrence matrix
        cooccurrence = (dtm.T * dtm).toarray()
        
        # Find significant co-occurrences
        for i, concept1 in enumerate(concepts):
            if concept1 not in concept_indices:
                continue
                
            idx1 = concept_indices[concept1]
            for j, concept2 in enumerate(concepts):
                if concept2 not in concept_indices:
                    continue
                    
                idx2 = concept_indices[concept2]
                if idx1 != idx2 and cooccurrence[idx1, idx2] > 0:
                    # Add bidirectional relationship
                    relationships.append((concept1, 'related_to', concept2))
                    relationships.append((concept2, 'related_to', concept1))
        
        return relationships

    def build_concept_graph(self, concepts: List[str], relationships: List[Tuple[str, str, str]]) -> nx.DiGraph:
        """Build a directed graph representation of the concept map."""
        G = nx.DiGraph()
        
        # Add nodes (concepts)
        for concept in concepts:
            G.add_node(concept)
        
        # Add edges (relationships)
        for concept1, rel_type, concept2 in relationships:
            G.add_edge(concept1, concept2, relationship=rel_type)
        
        return G

# Example usage
if __name__ == "__main__":
    # Sample texts
    sample_texts = [
        "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
        "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs."
    ]
    
    # Sample concepts
    sample_concepts = [
        "machine learning",
        "deep learning",
        "supervised learning",
        "neural networks",
        "input",
        "output",
        "function"
    ]
    
    # Create extractor
    extractor = RelationshipExtractor()
    
    # Extract relationships
    relationships = extractor.extract_relationships(sample_texts, sample_concepts)
    
    # Build concept graph
    G = extractor.build_concept_graph(sample_concepts, relationships)
    
    print("Extracted relationships:")
    for rel in relationships:
        print(f"{rel[0]} --[{rel[1]}]--> {rel[2]}") 