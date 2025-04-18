import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
import re
import random
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

    #def extract_relationships(self, texts: List[str], concepts: List[str]) -> List[Tuple[str, str, str]]:
    #def extract_relationships(self, texts: List[str], concepts: List[str], limit: int = 2000) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between concepts using both pattern matching and co-occurrence analysis.
        
        Args:
            texts: List of text segments to analyze
            concepts: List of concepts to find relationships between
            
        Returns:
            List of tuples (concept1, relationship_type, concept2)
        
        relationships = []
        
        # Pattern-based relationship extraction
        pattern_relationships = self._extract_pattern_relationships(texts, concepts)
        relationships.extend(pattern_relationships)
        
        # Co-occurrence based relationship extraction
        cooccurrence_relationships = self._extract_cooccurrence_relationships(texts, concepts)
        relationships.extend(cooccurrence_relationships)
        
        return relationships
        """
        """
        Extract relationships between concepts using both pattern matching and co-occurrence analysis,
        limited to a specified number of relationships with even distribution across patterns.
        
        Args:
            texts: List of text segments to analyze
            concepts: List of concepts to find relationships between
            limit: Maximum number of relationships to return (default: 100)
            
        Returns:
            List of tuples (concept1, relationship_type, concept2)
        
        # Pattern-based relationship extraction
        pattern_relationships_by_type = self._extract_pattern_relationships_by_type(texts, concepts)
        
        # Co-occurrence based relationship extraction
        cooccurrence_relationships = self._extract_cooccurrence_relationships(texts, concepts)
        
        # Combine and limit results with even distribution
        return self._limit_relationships_with_even_distribution(
            pattern_relationships_by_type, 
            cooccurrence_relationships, 
            limit
        """
        #)
    def extract_relationships(self, texts: List[str], concepts: List[str], limit: int = 100) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between concepts using both pattern matching and co-occurrence analysis,
        limited to a specified number of relationships with even distribution across patterns.
        
        Args:
            texts: List of text segments to analyze
            concepts: List of concepts to find relationships between
            limit: Maximum number of relationships to return (default: 100)
            
        Returns:
            List of tuples (concept1, relationship_type, concept2)
        """
        # Extract all relationships first
        all_relationships = []
        relationships_by_type = defaultdict(list)
        
        # Pattern-based relationship extraction
        for text in texts:
            for rel_type, patterns in self.relationship_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        concept1, concept2 = match.groups()
                        # Check if both concepts are in the provided concepts list (case-insensitive)
                        concept1_matches = [c for c in concepts if c.lower() == concept1.lower()]
                        concept2_matches = [c for c in concepts if c.lower() == concept2.lower()]
                        
                        if concept1_matches and concept2_matches:
                            relationship = (concept1_matches[0], rel_type, concept2_matches[0])
                            relationships_by_type[rel_type].append(relationship)
                            all_relationships.append(relationship)
        
        # Co-occurrence based relationship extraction (if needed)
        if 'related_to' not in relationships_by_type:
            cooccurrence_relationships = self._extract_cooccurrence_relationships(texts, concepts)
            relationships_by_type['related_to'] = cooccurrence_relationships
            all_relationships.extend(cooccurrence_relationships)
        
        # Apply limit with even distribution across pattern types
        return self._limit_with_even_distribution(relationships_by_type, limit)

    # New Code
    def _extract_pattern_relationships_by_type(self, texts: List[str], concepts: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
        """Extract relationships by pattern type"""
        relationships_by_type = {rel_type: [] for rel_type in self.relationship_patterns.keys()}
        
        # Create a set of concepts for faster lookup
        concept_set = set(concepts)
        
        for text in texts:
            for rel_type, patterns in self.relationship_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        concept1, concept2 = match.groups()
                        if concept1.lower() in [c.lower() for c in concept_set] and concept2.lower() in [c.lower() for c in concept_set]:
                            # Find the exact concept names with matching case
                            for c1 in concepts:
                                if c1.lower() == concept1.lower():
                                    for c2 in concepts:
                                        if c2.lower() == concept2.lower():
                                            relationships_by_type[rel_type].append((c1, rel_type, c2))
        
        return relationships_by_type
    
    def _extract_cooccurrence_relationships(self, texts: List[str], concepts: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships based on co-occurrence"""
        # This is a placeholder for your existing implementation
        # For demonstration, returning an empty list
        return []
    
    def _limit_with_even_distribution(self, relationships_by_type: Dict[str, List[Tuple[str, str, str]]], limit: int) -> List[Tuple[str, str, str]]:
        """
        Limit relationships with even distribution across relationship types.
        
        Args:
            relationships_by_type: Dictionary mapping relationship types to lists of relationships
            limit: Maximum number of relationships to return
            
        Returns:
            Limited list of relationships with even distribution
        """
        result = []
        
        # Filter out empty relationship types
        non_empty_types = {rel_type: rels for rel_type, rels in relationships_by_type.items() if rels}
        
        if not non_empty_types:
            return []
        
        # Calculate quota per relationship type
        num_types = len(non_empty_types)
        base_quota = limit // num_types
        remainder = limit % num_types
        
        # Sort relationship types to ensure deterministic behavior
        sorted_types = sorted(non_empty_types.keys())
        
        # Distribute relationships evenly
        for rel_type in sorted_types:
            relationships = non_empty_types[rel_type]
            # Shuffle to get a random sample if we're limiting
            random.shuffle(relationships)
            
            # Calculate quota for this type
            type_quota = base_quota + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
                
            # Add relationships up to the quota
            result.extend(relationships[:type_quota])
        
        return result



    # Old code
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