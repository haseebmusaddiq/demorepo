import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for each stage
colors = {
    'indexing': '#3498db',
    'query': '#2ecc71',
    'rerank': '#e74c3c',
    'context': '#f39c12',
    'generation': '#9b59b6',
    'presentation': '#1abc9c',
    'database': '#34495e',
    'config': '#95a5a6'
}

# Function to create a box with title and steps
def create_box(x, y, width, height, title, steps, color):
    # Main box
    box = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', 
                           facecolor=color, alpha=0.7)
    ax.add_patch(box)
    
    # Title
    ax.text(x + width/2, y + height - 0.2, title, 
            horizontalalignment='center', fontsize=12, fontweight='bold', color='white')
    
    # Steps
    for i, step in enumerate(steps):
        ax.text(x + 0.1, y + height - 0.5 - i*0.4, f"• {step}", 
                fontsize=9, color='white')

# Create the workflow boxes
# 1. Document Indexing
create_box(0, 5, 3, 2.5, "1. Document Indexing", 
          ["load_documents()", "Split into chunks", "generate_embeddings()", "add_documents()"], 
          colors['indexing'])

# 2. Query Processing
create_box(4, 5, 3, 2.5, "2. Query Processing", 
          ["API Endpoint", "generate_query_embedding()", "hybrid_search()", "Retrieve top_k"], 
          colors['query'])

# 3. Reranking
create_box(8, 5, 3, 2.5, "3. Reranking", 
          ["rerank_documents()", "cross_encoder.predict()", "Calculate final_score", "Sort by score"], 
          colors['rerank'])

# 4. Context Building
create_box(0, 1.5, 3, 2.5, "4. Context Building", 
          ["Format document chunks", "Create context strings", "build_prompt()"], 
          colors['context'])

# 5. Response Generation
create_box(4, 1.5, 3, 2.5, "5. Response Generation", 
          ["generate_response()", "Select LLM provider", "Process response"], 
          colors['generation'])

# 6. Result Presentation
create_box(8, 1.5, 3, 2.5, "6. Result Presentation", 
          ["Format JSON response", "Include source info", "Return jsonify response"], 
          colors['presentation'])

# Database and Config
db = patches.Rectangle((4, 0, 1), 1, 1, linewidth=1, edgecolor='black', 
                      facecolor=colors['database'], alpha=0.7)
ax.add_patch(db)
ax.text(4.5, 0.5, "Vector DB", horizontalalignment='center', fontsize=10, color='white')

config = patches.Rectangle((6, 0, 1), 1, 1, linewidth=1, edgecolor='black', 
                          facecolor=colors['config'], alpha=0.7)
ax.add_patch(config)
ax.text(6.5, 0.5, "config.yaml", horizontalalignment='center', fontsize=10, color='black')

# Add arrows connecting the workflow stages
def add_arrow(x1, y1, x2, y2, color='black', style='-'):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, linestyle=style))

# Main workflow arrows
add_arrow(3, 6.5, 4, 6.5)  # Indexing to Query
add_arrow(7, 6.5, 8, 6.5)  # Query to Reranking
add_arrow(9.5, 5, 1.5, 4)  # Reranking to Context
add_arrow(3, 2.5, 4, 2.5)  # Context to Generation
add_arrow(7, 2.5, 8, 2.5)  # Generation to Presentation

# Database interactions
add_arrow(1.5, 5, 4.5, 1, color='blue', style='--')  # Indexing to DB
add_arrow(4.5, 1, 5.5, 5, color='blue', style='--')  # DB to Query

# Config influences
add_arrow(6.5, 1, 5.5, 5, color='green', style=':')  # Config to Query
add_arrow(6.5, 1, 9.5, 5, color='green', style=':')  # Config to Reranking
add_arrow(6.5, 1, 1.5, 1.5, color='green', style=':')  # Config to Context
add_arrow(6.5, 1, 5.5, 1.5, color='green', style=':')  # Config to Generation

# Set axis limits and remove ticks
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)
ax.set_xticks([])
ax.set_yticks([])

# Add title
plt.title('RAG System Workflow', fontsize=16, fontweight='bold')

# Save the figure
plt.savefig('rag_workflow_diagram.png', dpi=300, bbox_inches='tight')
print("Diagram created: rag_workflow_diagram.png")

# Show the plot
plt.show()