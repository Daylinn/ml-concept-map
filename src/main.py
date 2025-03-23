import os
import sys
from typing import List, Tuple

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.concept_extraction.basic_extractor import ConceptExtractor
from src.models.relationship_extraction.relationship_extractor import RelationshipExtractor
from src.visualization.concept_map_visualizer import ConceptMapVisualizer

def load_text_from_file(file_path: str) -> str:
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into smaller chunks for processing."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += 1
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def main():
    # Initialize components
    concept_extractor = ConceptExtractor()
    relationship_extractor = RelationshipExtractor()
    visualizer = ConceptMapVisualizer()
    
    # Load and process text
    input_file = "data/input.txt"  # You'll need to provide your input text file
    text = load_text_from_file(input_file)
    chunks = split_into_chunks(text)
    
    # Extract concepts
    concepts = concept_extractor.extract_all_concepts(chunks, top_n=50)
    print(f"Extracted {len(concepts)} concepts")
    
    # Extract relationships
    relationships = relationship_extractor.extract_relationships(chunks, concepts)
    print(f"Extracted {len(relationships)} relationships")
    
    # Build concept graph
    G = relationship_extractor.build_concept_graph(concepts, relationships)
    
    # Create visualizations
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Static visualization
    visualizer.visualize_graph(G, 
                             title="ML Concept Map",
                             save_path=os.path.join(output_dir, "concept_map.png"))
    
    # Interactive visualization
    visualizer.create_interactive_html(G, 
                                     os.path.join(output_dir, "concept_map.html"))
    
    print(f"Visualizations saved in the '{output_dir}' directory")

if __name__ == "__main__":
    main() 