import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import numpy as np

class ConceptMapVisualizer:
    def __init__(self):
        plt.style.use('default')
        self.colors = {
            'is_a': '#FF9999',
            'has_part': '#99FF99',
            'used_for': '#9999FF',
            'related_to': '#FFCC99'
        }
        
    def visualize_graph(self, 
                       G: nx.DiGraph,
                       title: str = "Concept Map",
                       figsize: tuple = (12, 8),
                       save_path: Optional[str] = None) -> None:
        """
        Visualize the concept map graph.
        
        Args:
            G: NetworkX directed graph
            title: Title for the visualization
            figsize: Figure size (width, height)
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=figsize)
        
        # Set up the layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=2000,
                             alpha=0.7)
        
        # Draw edges with different colors based on relationship type
        edge_colors = [self.colors.get(G[u][v]['relationship'], 'gray') 
                      for u, v in G.edges()]
        
        nx.draw_networkx_edges(G, pos,
                             edge_color=edge_colors,
                             arrows=True,
                             arrowsize=20)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos,
                              font_size=8,
                              font_weight='bold')
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'relationship')
        nx.draw_networkx_edge_labels(G, pos,
                                   edge_labels=edge_labels,
                                   font_size=6)
        
        plt.title(title, pad=20)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def create_interactive_html(self, 
                              G: nx.DiGraph,
                              output_path: str) -> None:
        """
        Create an interactive HTML visualization using D3.js.
        
        Args:
            G: NetworkX directed graph
            output_path: Path to save the HTML file
        """
        # Convert NetworkX graph to JSON format
        nodes = [{'id': node, 'group': 1} for node in G.nodes()]
        links = [{'source': u, 
                 'target': v, 
                 'type': G[u][v]['relationship']} 
                for u, v in G.edges()]
        
        # Create HTML template
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive Concept Map</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body {{ margin: 0; overflow: hidden; }}
                #graph {{ width: 100vw; height: 100vh; }}
                .node {{ cursor: pointer; }}
                .link {{ stroke: #999; stroke-opacity: 0.6; }}
                .label {{ font-size: 12px; }}
            </style>
        </head>
        <body>
            <div id="graph"></div>
            <script>
                const data = {{
                    nodes: {nodes},
                    links: {links}
                }};
                
                const width = window.innerWidth;
                const height = window.innerHeight;
                
                const svg = d3.select("#graph")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                const simulation = d3.forceSimulation(data.nodes)
                    .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
                    .force("charge", d3.forceManyBody().strength(-300))
                    .force("center", d3.forceCenter(width / 2, height / 2));
                
                const link = svg.append("g")
                    .selectAll("line")
                    .data(data.links)
                    .join("line")
                    .attr("class", "link");
                
                const node = svg.append("g")
                    .selectAll("g")
                    .data(data.nodes)
                    .join("g")
                    .attr("class", "node")
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                
                node.append("circle")
                    .attr("r", 5)
                    .style("fill", "#69b3a2");
                
                node.append("text")
                    .attr("class", "label")
                    .attr("dx", 12)
                    .attr("dy", ".35em")
                    .text(d => d.id);
                
                simulation.on("tick", () => {{
                    link
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    node
                        .attr("transform", d => `translate(${{d.x}}, ${{d.y}})`);
                }});
                
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}
                
                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}
                
                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}
            </script>
        </body>
        </html>
        """
        
        # Save the HTML file
        with open(output_path, 'w') as f:
            f.write(html_template)

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
    
    # Create visualizer
    visualizer = ConceptMapVisualizer()
    
    # Create static visualization
    visualizer.visualize_graph(G, "Sample Concept Map")
    
    # Create interactive visualization
    visualizer.create_interactive_html(G, "concept_map.html") 