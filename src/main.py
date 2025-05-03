# import os
# import sys
# from pathlib import Path
# from typing import List

# # Add the src directory to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.models.concept_extraction.basic_extractor import ConceptExtractor
# from src.models.relationship_extraction.relationship_extractor import RelationshipExtractor
# from src.visualization.concept_map_visualizer import ConceptMapVisualizer

# def load_text_from_file(file_path: Path) -> str:
#     """Load text from a file with comprehensive error handling."""
#     try:
#         return file_path.read_text(encoding='utf-8')
#     except FileNotFoundError:
#         print(f"\n✖ Error: File not found at {file_path}")
#         print("Please verify:")
#         print(f"- The file exists at this location")
#         print(f"- The filename is correct (check case sensitivity)")
#         print(f"- No hidden file extensions (.txt vs .txt.txt)")
#         sys.exit(1)
#     except Exception as e:
#         print(f"\n✖ Error reading file: {e}")
#         sys.exit(1)

# def load_chapters(chapter_nums: List[int], split: str = 'train') -> str:
#     """Load multiple chapter texts with robust path handling."""
#     chapters = []
#     # Using Path for reliable path handling across operating systems
#     base_path = Path(__file__).parent.parent / "data" / "processed" / split
    
#     print(f"\nLoading {split} chapters from: {base_path}")
    
#     for num in chapter_nums:
#         # Try multiple possible filename patterns
#         possible_filenames = [
#             f"ch{num}.txt",          # Your actual format
#             f"chapter_{num}.txt",     # Alternative format
#             f"Ch{num}.txt",           # Case variation
#             f"chapter{num}.txt"       # No underscore
#         ]
        
#         found = False
#         for filename in possible_filenames:
#             file_path = base_path / filename
#             if file_path.exists():
#                 chapters.append(load_text_from_file(file_path))
#                 found = True
#                 break
        
#         if not found:
#             print(f"\n✖ Could not find chapter {num} in {split} data")
#             print(f"Tried: {', '.join(possible_filenames)}")
#             sys.exit(1)
    
#     return "\n".join(chapters)

# def split_into_chunks(text: str, chunk_size: int = 1000) -> List[str]:
#     """Split text into smaller chunks for processing."""
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_size = 0
    
#     for word in words:
#         current_chunk.append(word)
#         current_size += 1
#         if current_size >= chunk_size:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = []
#             current_size = 0
    
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
    
#     return chunks

# def main():
#     # Initialize components
#     concept_extractor = ConceptExtractor()
#     relationship_extractor = RelationshipExtractor()
#     visualizer = ConceptMapVisualizer()
    
#     # Define train/test split
#     train_chapters = [1,2,3,4,5,7,8,9,13,14,15,16,17,18,19]
#     test_chapters = [6,10,11,12]
    
#     # Load and process training data
#     print("Loading and processing training data...")
#     train_text = load_chapters(train_chapters, split='train')
#     train_chunks = split_into_chunks(train_text)
    
#     # Extract concepts from training data
#     print("\nExtracting concepts from training data...")
#     train_concepts = concept_extractor.extract_all_concepts(train_chunks, top_n=12)
#     print(f"Extracted {len(train_concepts)} training concepts")
    
#     # Load and process test data
#     print("\nLoading and processing test data...")
#     test_text = load_chapters(test_chapters, split='test')
#     test_chunks = split_into_chunks(test_text)
    
#     # Extract concepts from test data (for evaluation only)
#     test_concepts = concept_extractor.extract_all_concepts(test_chunks, top_n=12)
#     print(f"Extracted {len(test_concepts)} test concepts")
    
#     # Extract relationships from training data only
#     print("\nExtracting relationships from training data...")
#     relationships = relationship_extractor.extract_relationships(
#         train_chunks, 
#         train_concepts, 
#         limit=200
#     )
#     print(f"Extracted {len(relationships)} relationships")
    
#     # Build and visualize concept graph
#     G = relationship_extractor.build_concept_graph(train_concepts, relationships)
    
#     # Create visualizations
#     output_dir = Path(__file__).parent.parent / "output"
#     output_dir.mkdir(exist_ok=True)
    
#     print("\nGenerating visualizations...")
#     visualizer.visualize_graph(
#         G, 
#         title="Data Magicians - Concept Map (Train Data)",
#         save_path=str(output_dir / "concept_map.png")
#     )
    
#     visualizer.create_interactive_html(
#         G, 
#         str(output_dir / "concept_map.html")
#     )

# # ... (keep all the previous imports and helper functions)

# def generate_concept_map(extractor: ConceptExtractor, 
#                         relationship_extractor: RelationshipExtractor,
#                         visualizer: ConceptMapVisualizer,
#                         chapters: List[int], 
#                         split: str,
#                         output_dir: Path,
#                         title_suffix: str) -> None:
#     """Generate a concept map for specific chapters."""
#     print(f"\nGenerating concept map for {split} data (chapters {chapters})...")
#     text = load_chapters(chapters, split=split)
#     chunks = split_into_chunks(text)
    
#     # Extract concepts
#     concepts = extractor.extract_all_concepts(chunks, top_n=12)
#     print(f"Extracted {len(concepts)} concepts")
    
#     # Extract relationships
#     relationships = relationship_extractor.extract_relationships(chunks, concepts, limit=200)
#     print(f"Extracted {len(relationships)} relationships")
    
#     # Build graph
#     G = relationship_extractor.build_concept_graph(concepts, relationships)
    
#     # Create visualizations
#     base_filename = f"concept_map_{split}"
#     visualizer.visualize_graph(
#         G, 
#         title=f"Data Magicians - Concept Map - {title_suffix}",
#         save_path=str(output_dir / f"{base_filename}.png")
#     )
    
#     visualizer.create_interactive_html(
#         G, 
#         str(output_dir / f"{base_filename}.html")
#     )

# def main():
#     # Initialize components
#     concept_extractor = ConceptExtractor()
#     relationship_extractor = RelationshipExtractor()
#     visualizer = ConceptMapVisualizer()
    
#     # Define train/test split
#     train_chapters = [1,2,3,4,5,7,8,9,13,14,15,16,17,18,19]
#     test_chapters = [6,10,11,12]
    
#     # Setup output directory
#     output_dir = Path(__file__).parent.parent / "output"
#     output_dir.mkdir(exist_ok=True)
    
#     # Generate training data concept map
#     generate_concept_map(
#         concept_extractor,
#         relationship_extractor,
#         visualizer,
#         train_chapters,
#         split='train',
#         output_dir=output_dir,
#         title_suffix="Training Data"
#     )
    
#     # Generate test data concept map
#     generate_concept_map(
#         concept_extractor,
#         relationship_extractor,
#         visualizer,
#         test_chapters,
#         split='test',
#         output_dir=output_dir,
#         title_suffix="Test Data"
#     )
    
#     print(f"\nAll visualizations saved in '{output_dir}' directory:")
#     print(f"- concept_map_train.png/html (Training chapters: {train_chapters})")
#     print(f"- concept_map_test.png/html (Test chapters: {test_chapters})")

# if __name__ == "__main__":
#     main()
    
# #     print(f"\nVisualizations saved in '{output_dir}' directory")

# # if __name__ == "__main__":
# #     main()


import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import networkx as nx

class ConceptExtractor:
    """Basic concept extractor (simplified for this example)"""
    def extract_all_concepts(self, chunks: List[str], top_n: int = 12) -> List[str]:
        """Mock concept extraction - in real implementation, this would use NLP"""
        # This is a simplified version - real implementation would do actual NLP processing
        mock_concepts = ["data", "analysis", "machine learning", "statistics", 
                        "visualization", "python", "algorithm", "model",
                        "probability", "regression", "classification", "AI"]
        return mock_concepts[:top_n]

class RelationshipExtractor:
    """Basic relationship extractor (simplified for this example)"""
    def extract_relationships(self, chunks: List[str], concepts: List[str], limit: int = 200) -> List[tuple]:
        """Mock relationship extraction - in real implementation, this would analyze text"""
        # Create some mock relationships
        relationships = [
            ("data", "analysis", 5),
            ("machine learning", "algorithm", 4),
            ("statistics", "probability", 3),
            ("visualization", "python", 3),
            ("regression", "statistics", 4),
            ("classification", "machine learning", 4),
            ("AI", "machine learning", 5),
            ("model", "data", 3)
        ]
        return relationships[:limit]
    
    def build_concept_graph(self, concepts: List[str], relationships: List[tuple]) -> nx.Graph:
        """Build a networkx graph from concepts and relationships"""
        G = nx.Graph()
        G.add_nodes_from(concepts)
        for src, dest, weight in relationships:
            G.add_edge(src, dest, weight=weight)
        return G

class ConceptMapVisualizer:
    """Visualizer for concept maps"""
    def visualize_graph(self, G: nx.Graph, title: str, save_path: str) -> None:
        """Visualize the graph and save to file"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold')
        plt.title(title)
        plt.savefig(save_path)
        plt.close()
    
    def create_interactive_html(self, G: nx.Graph, save_path: str) -> None:
        """Create an interactive HTML visualization (simplified for this example)"""
        # In a real implementation, this would use pyvis or similar
        with open(save_path, 'w') as f:
            f.write("<html><body><h1>Concept Map</h1><p>Interactive visualization would appear here</p></body></html>")

class ConceptAnalyzer:
    """Class to perform statistical analysis on concept relationships"""
    
    def __init__(self):
        self.regression_model = LinearRegression()
    
    def analyze_concept_frequency_trend(self, concept_counts: dict) -> Dict[str, Dict[str, float]]:
        """
        Analyze trend of concept frequencies across chapters using linear regression.
        
        Args:
            concept_counts: Dictionary {concept: {chapter: count}}
        
        Returns:
            Dictionary with regression results for each concept
        """
        results = {}
        
        for concept, chapter_counts in concept_counts.items():
            # Prepare data
            chapters = np.array(sorted(chapter_counts.keys())).reshape(-1, 1)
            counts = np.array([chapter_counts[ch] for ch in sorted(chapter_counts.keys())])
            
            # Fit linear regression
            self.regression_model.fit(chapters, counts)
            
            # Store results
            results[concept] = {
                'slope': self.regression_model.coef_[0],
                'intercept': self.regression_model.intercept_,
                'r_squared': self.regression_model.score(chapters, counts)
            }
            
        return results
    
    def plot_concept_trend(self, concept: str, chapter_counts: dict, save_path: str = None) -> None:
        """
        Plot concept frequency trend with regression line.
        
        Args:
            concept: Concept name
            chapter_counts: Dictionary {chapter: count}
            save_path: Path to save the plot (optional)
        """
        chapters = np.array(sorted(chapter_counts.keys())).reshape(-1, 1)
        counts = np.array([chapter_counts[ch] for ch in sorted(chapter_counts.keys())])
        
        # Fit model
        self.regression_model.fit(chapters, counts)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(chapters, counts, color='blue', label='Actual frequency')
        
        # Plot regression line
        predicted = self.regression_model.predict(chapters)
        plt.plot(chapters, predicted, color='red', 
                label=f'Linear trend (R²={self.regression_model.score(chapters, counts):.2f})')
        
        plt.title(f'Frequency Trend for Concept: {concept}')
        plt.xlabel('Chapter Number')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def load_text_from_file(file_path: Path) -> str:
    """Load text from a file with comprehensive error handling."""
    try:
        return file_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"\n✖ Error: File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✖ Error reading file: {e}")
        sys.exit(1)

def load_chapters(chapter_nums: List[int], split: str = 'train') -> str:
    """Load multiple chapter texts with robust path handling."""
    chapters = []
    base_path = Path(__file__).parent.parent / "data" / "processed" / split
    
    print(f"\nLoading {split} chapters from: {base_path}")
    
    for num in chapter_nums:
        possible_filenames = [
            f"ch{num}.txt",          
            f"chapter_{num}.txt",     
            f"Ch{num}.txt",           
            f"chapter{num}.txt"       
        ]
        
        found = False
        for filename in possible_filenames:
            file_path = base_path / filename
            if file_path.exists():
                chapters.append(load_text_from_file(file_path))
                found = True
                break
        
        if not found:
            print(f"\n✖ Could not find chapter {num} in {split} data")
            print(f"Tried: {', '.join(possible_filenames)}")
            sys.exit(1)
    
    return "\n".join(chapters)

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

def count_concepts_by_chapter(extractor: ConceptExtractor, 
                            chunks: List[str], 
                            chapter_nums: List[int]) -> Dict[str, Dict[int, int]]:
    """
    Count concept occurrences by chapter.
    
    Args:
        extractor: Concept extractor instance
        chunks: List of text chunks
        chapter_nums: List of chapter numbers
        
    Returns:
        Dictionary {concept: {chapter: count}}
    """
    # For this example, we'll simulate some data
    # In a real implementation, you would:
    # 1. Know which chunks belong to which chapter
    # 2. Count actual concept occurrences
    
    concept_counts = {}
    concepts = extractor.extract_all_concepts(chunks)
    
    # Create mock data for demonstration
    for concept in concepts:
        chapter_counts = {}
        for chapter in chapter_nums:
            # Simulate some frequency pattern
            if "data" in concept:
                chapter_counts[chapter] = np.random.randint(5, 20) + chapter
            elif "machine" in concept:
                chapter_counts[chapter] = np.random.randint(3, 15) + chapter*2
            else:
                chapter_counts[chapter] = np.random.randint(1, 10) + chapter//2
        concept_counts[concept] = chapter_counts
    
    return concept_counts

def generate_concept_map(extractor: ConceptExtractor, 
                        relationship_extractor: RelationshipExtractor,
                        visualizer: ConceptMapVisualizer,
                        chapters: List[int], 
                        split: str,
                        output_dir: Path,
                        title_suffix: str) -> None:
    """Generate a concept map for specific chapters."""
    print(f"\nGenerating concept map for {split} data (chapters {chapters})...")
    text = load_chapters(chapters, split=split)
    chunks = split_into_chunks(text)
    
    # Extract concepts
    concepts = extractor.extract_all_concepts(chunks, top_n=12)
    print(f"Extracted {len(concepts)} concepts")
    
    # Extract relationships
    relationships = relationship_extractor.extract_relationships(chunks, concepts, limit=200)
    print(f"Extracted {len(relationships)} relationships")
    
    # Build graph
    G = relationship_extractor.build_concept_graph(concepts, relationships)
    
    # Create visualizations
    base_filename = f"concept_map_{split}"
    visualizer.visualize_graph(
        G, 
        title=f"Concept Map - {title_suffix}",
        save_path=str(output_dir / f"{base_filename}.png")
    )
    visualizer.create_interactive_html(
        G, 
        str(output_dir / f"{base_filename}.html"))

def main():
    # Initialize components
    concept_extractor = ConceptExtractor()
    relationship_extractor = RelationshipExtractor()
    visualizer = ConceptMapVisualizer()
    analyzer = ConceptAnalyzer()
    
    # Define train/test split
    train_chapters = [1,2,3,4,5,7,8,9,13,14,15,16,17,18,19]
    test_chapters = [6,10,11,12]
    
    # Setup output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Generate training data concept map
    generate_concept_map(
        concept_extractor,
        relationship_extractor,
        visualizer,
        train_chapters,
        split='train',
        output_dir=output_dir,
        title_suffix="Training Data"
    )
    
    # Generate test data concept map
    generate_concept_map(
        concept_extractor,
        relationship_extractor,
        visualizer,
        test_chapters,
        split='test',
        output_dir=output_dir,
        title_suffix="Test Data"
    )
    
    # Perform concept frequency analysis
    print("\nAnalyzing concept trends...")
    
    # Load training data again for analysis
    train_text = load_chapters(train_chapters, split='train')
    train_chunks = split_into_chunks(train_text)
    
    # Get concept counts by chapter
    concept_counts = count_concepts_by_chapter(concept_extractor, train_chunks, train_chapters)
    
    # Analyze trends
    trend_results = analyzer.analyze_concept_frequency_trend(concept_counts)
    
    # Print top concepts with strongest trends
    sorted_concepts = sorted(trend_results.items(), 
                           key=lambda x: abs(x[1]['slope']), 
                           reverse=True)[:5]
    
    print("\nTop concepts with strongest trends:")
    for concept, stats in sorted_concepts:
        print(f"{concept}: slope={stats['slope']:.2f} (R²={stats['r_squared']:.2f})")
        analyzer.plot_concept_trend(
            concept, 
            concept_counts[concept],
            save_path=str(output_dir / f"trend_{concept.lower().replace(' ', '_')}.png")
        )
    
    print(f"\nAll visualizations saved in '{output_dir}' directory")

if __name__ == "__main__":
    main()
