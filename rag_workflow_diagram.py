import graphviz

# Create a new directed graph
dot = graphviz.Digraph('RAG_Workflow', comment='RAG System Workflow')

# Set graph attributes
dot.attr(rankdir='TB', size='11,8', dpi='300')
dot.attr('node', shape='box', style='filled,rounded', fontname='Arial', fontsize='12')
dot.attr('edge', fontname='Arial', fontsize='10')

# Define node colors
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

# Create clusters for each workflow stage
with dot.subgraph(name='cluster_1') as c:
    c.attr(label='1. Document Indexing', style='filled', color=colors['indexing'], fontcolor='white')
    c.node('A1', 'doc_processor.load_documents()', style='filled', fillcolor='#2980b9', fontcolor='white')
    c.node('A2', 'Split into chunks', style='filled', fillcolor='#2980b9', fontcolor='white')
    c.node('A3', 'embedding_manager.generate_embeddings()', style='filled', fillcolor='#2980b9', fontcolor='white')
    c.node('A4', 'vector_store.add_documents()', style='filled', fillcolor='#2980b9', fontcolor='white')
    c.edges([('A1', 'A2'), ('A2', 'A3'), ('A3', 'A4')])

with dot.subgraph(name='cluster_2') as c:
    c.attr(label='2. Query Processing', style='filled', color=colors['query'], fontcolor='white')
    c.node('B1', 'app.route(\'/query\')', style='filled', fillcolor='#27ae60', fontcolor='white')
    c.node('B2', 'embedding_manager.generate_query_embedding()', style='filled', fillcolor='#27ae60', fontcolor='white')
    c.node('B3', 'vector_store.hybrid_search()', style='filled', fillcolor='#27ae60', fontcolor='white')
    c.node('B4', 'Retrieve top_k documents', style='filled', fillcolor='#27ae60', fontcolor='white')
    c.edges([('B1', 'B2'), ('B2', 'B3'), ('B3', 'B4')])

with dot.subgraph(name='cluster_3') as c:
    c.attr(label='3. Reranking', style='filled', color=colors['rerank'], fontcolor='white')
    c.node('C1', 'rerank_documents()', style='filled', fillcolor='#c0392b', fontcolor='white')
    c.node('C2', 'cross_encoder.predict()', style='filled', fillcolor='#c0392b', fontcolor='white')
    c.node('C3', 'Calculate final_score', style='filled', fillcolor='#c0392b', fontcolor='white')
    c.node('C4', 'Sort documents by score', style='filled', fillcolor='#c0392b', fontcolor='white')
    c.edges([('C1', 'C2'), ('C2', 'C3'), ('C3', 'C4')])

with dot.subgraph(name='cluster_4') as c:
    c.attr(label='4. Context Building', style='filled', color=colors['context'], fontcolor='white')
    c.node('D1', 'Format document chunks', style='filled', fillcolor='#d35400', fontcolor='white')
    c.node('D2', 'Create context strings with file_path', style='filled', fillcolor='#d35400', fontcolor='white')
    c.node('D3', 'llm_manager.build_prompt()', style='filled', fillcolor='#d35400', fontcolor='white')
    c.edges([('D1', 'D2'), ('D2', 'D3')])

with dot.subgraph(name='cluster_5') as c:
    c.attr(label='5. Response Generation', style='filled', color=colors['generation'], fontcolor='white')
    c.node('E1', 'llm_manager.generate_response()', style='filled', fillcolor='#8e44ad', fontcolor='white')
    c.node('E2', 'Select LLM provider', style='filled', fillcolor='#8e44ad', fontcolor='white')
    c.node('E3', 'Process response', style='filled', fillcolor='#8e44ad', fontcolor='white')
    c.edges([('E1', 'E2'), ('E2', 'E3')])

with dot.subgraph(name='cluster_6') as c:
    c.attr(label='6. Result Presentation', style='filled', color=colors['presentation'], fontcolor='white')
    c.node('F1', 'Format JSON response', style='filled', fillcolor='#16a085', fontcolor='white')
    c.node('F2', 'Include source information', style='filled', fillcolor='#16a085', fontcolor='white')
    c.node('F3', 'Return jsonify response', style='filled', fillcolor='#16a085', fontcolor='white')
    c.edges([('F1', 'F2'), ('F2', 'F3')])

# Add database and config nodes
dot.node('DB', 'Vector Database', shape='cylinder', style='filled', fillcolor=colors['database'], fontcolor='white')
dot.node('Config', 'config.yaml', shape='note', style='filled', fillcolor=colors['config'], fontcolor='white')

# Connect the workflow stages
dot.edges([('A4', 'B1'), ('B4', 'C1'), ('C4', 'D1'), ('D3', 'E1'), ('E3', 'F1')])

# Add data flow edges
dot.edge('A4', 'DB', style='dashed', label='Store embeddings')
dot.edge('DB', 'B3', style='dashed', label='Query database')

# Add configuration influences
dot.edge('Config', 'B3', style='dashed', label='top_k, alpha')
dot.edge('Config', 'C3', style='dashed', label='rerank_weight')
dot.edge('Config', 'D3', style='dashed', label='max_context_length')
dot.edge('Config', 'E2', style='dashed', label='provider selection')

# Render the graph
dot.render('rag_workflow_diagram', format='png', cleanup=True)
print("Diagram created: rag_workflow_diagram.png")

# If you want to view the diagram immediately (works on some systems)
import os
os.system('rag_workflow_diagram.png')