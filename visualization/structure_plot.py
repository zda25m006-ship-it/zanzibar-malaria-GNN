
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

def plot_gnn_structure(save_path='results/gnn_structure.png'):
    """
    Visualizes the GNN topology for Unguja districts.
    Nodes = Districts
    Edges = Adjacency (approximate based on geography)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Unguja Districts and approximate layout
    # (0,0) is bottom-left
    pos = {
        'North A': (0.3, 0.9),
        'North B': (0.7, 0.85),
        'Central': (0.6, 0.6),
        'West A':  (0.2, 0.55),
        'West B':  (0.2, 0.35),
        'Urban':   (0.1, 0.45),
        'South':   (0.5, 0.2),
    }
    
    # Adjacency (approximate)
    edges = [
        ('North A', 'North B'), ('North A', 'Central'), ('North A', 'West A'),
        ('North B', 'Central'), ('North B', 'South'), # Long edge?
        ('Central', 'South'), ('Central', 'West A'), ('Central', 'West B'),
        ('South', 'West B'),
        ('West A', 'Urban'), ('West A', 'West B'),
        ('Urban', 'West B'),
    ]
    
    G = nx.Graph()
    G.add_nodes_from(pos.keys())
    G.add_edges_from(edges)
    
    plt.figure(figsize=(8, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#4CAF50', alpha=0.9, edgecolors='white', linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='#666666')
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='white')
    
    plt.title("GNN Spatial Structure (Unguja Districts)", fontsize=15, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved GNN Structure: {save_path}")

if __name__ == "__main__":
    plot_gnn_structure()
