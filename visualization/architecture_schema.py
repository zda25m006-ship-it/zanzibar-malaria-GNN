
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_schema(save_path='results/gnn_architecture_schema.png'):
    """
    Draws a schematic of the GNN architecture showing:
    - Inputs (Risk Scores, Weather)
    - Spatial Aggregation (GAT)
    - Temporal Memory (GRU)
    - Output (Prediction)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Coordinates
    input_y = 0.1
    gat_y = 0.4
    gru_y = 0.7
    output_y = 0.9
    
    center_x = 0.5
    left_x = 0.2
    right_x = 0.8
    
    # --- 1. Input Layer ---
    # Box for Central Node Features
    ax.add_patch(patches.Rectangle((center_x-0.15, input_y), 0.3, 0.15, fill=True, color='#E3F2FD', ec='black'))
    ax.text(center_x, input_y+0.075, "Node Features\n(Risk Score, Rain, Cases t-1)", 
            ha='center', va='center', fontsize=10, fontweight='bold')
            
    # Neighbors
    ax.add_patch(patches.Rectangle((left_x-0.1, input_y), 0.2, 0.15, fill=True, color='#F1F8E9', ec='black'))
    ax.text(left_x, input_y+0.075, "Neighbor 1\nFeatures", ha='center', va='center', fontsize=9)
    
    ax.add_patch(patches.Rectangle((right_x-0.1, input_y), 0.2, 0.15, fill=True, color='#F1F8E9', ec='black'))
    ax.text(right_x, input_y+0.075, "Neighbor 2\nFeatures", ha='center', va='center', fontsize=9)
    
    # --- 2. GAT Layer (Spatial) ---
    # Central Aggregation
    ax.add_patch(patches.Circle((center_x, gat_y), 0.08, fill=True, color='#4CAF50', ec='black'))
    ax.text(center_x, gat_y, "GAT\nAgg", ha='center', va='center', color='white', fontweight='bold')
    
    # Arrows from Inputs to GAT
    ax.annotate("", xy=(center_x, gat_y-0.08), xytext=(center_x, input_y+0.15),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(center_x, gat_y-0.08), xytext=(left_x+0.1, input_y+0.15),
                arrowprops=dict(arrowstyle="->", lw=2, color='gray', linestyle='--'))
    ax.annotate("", xy=(center_x, gat_y-0.08), xytext=(right_x-0.1, input_y+0.15),
                arrowprops=dict(arrowstyle="->", lw=2, color='gray', linestyle='--'))
                
    ax.text(0.35, 0.3, "Attention Weights (α)", ha='center', fontsize=9, color='gray', style='italic')

    # --- 3. GRU Layer (Temporal) ---
    ax.add_patch(patches.Rectangle((center_x-0.1, gru_y-0.05), 0.2, 0.1, fill=True, color='#9C27B0', ec='black'))
    ax.text(center_x, gru_y, "GRU Memory\n(Update State)", ha='center', va='center', color='white', fontweight='bold')
    
    # Arrow from GAT to GRU
    ax.annotate("", xy=(center_x, gru_y-0.05), xytext=(center_x, gat_y+0.08),
                arrowprops=dict(arrowstyle="->", lw=2))
                
    # Recurrent Loop
    ax.annotate("", xy=(center_x+0.1, gru_y), xytext=(center_x+0.1, gru_y),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=2.5", lw=1.5))
    ax.text(center_x+0.22, gru_y, "History\n(t-1)", ha='center', va='center', fontsize=9)

    # --- 4. Output Layer ---
    ax.add_patch(patches.Rectangle((center_x-0.1, output_y-0.05), 0.2, 0.08, fill=True, color='#FFC107', ec='black'))
    ax.text(center_x, output_y-0.01, "Prediction (t)", ha='center', va='center', fontweight='bold')
    
    # Arrow from GRU to Output
    ax.annotate("", xy=(center_x, output_y-0.05), xytext=(center_x, gru_y+0.05),
                arrowprops=dict(arrowstyle="->", lw=2))

    # --- Titles ---
    plt.text(0.5, 0.02, "Input: Individual Features & Rainfall", ha='center', fontsize=11, fontweight='bold')
    plt.text(0.5, 0.98, "Output: Malaria Cases", ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved Architecture Schema: {save_path}")

if __name__ == "__main__":
    draw_architecture_schema()
