# app.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from data_processor import SynAestheticsDataProcessor
from chills_rag import ChillsRAGSystem
import chromadb

st.set_page_config(page_title="ChillsDB KG Explorer", layout="wide")

# Initialize session state
if "processor" not in st.session_state:
    with st.spinner("Loading ChillsDB data and building KG..."):
        st.session_state.processor = SynAestheticsDataProcessor()
        df, G = st.session_state.processor.run_pipeline()
        st.session_state.G = G
        st.session_state.rag = ChillsRAGSystem(graph=G, use_llm=False)
        st.session_state.messages = []

st.title("🧊 ChillsDB Knowledge Graph Explorer")

# Layout: KG visualization on top, chat below
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("3D Knowledge Graph")
    # Render the KG from the stored graph
    G = st.session_state.G
    
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = G.nodes[u]['pos']
        x1, y1, z1 = G.nodes[v]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    node_x = [G.nodes[n]['pos'][0] for n in G.nodes]
    node_y = [G.nodes[n]['pos'][1] for n in G.nodes]
    node_z = [G.nodes[n]['pos'][2] for n in G.nodes]
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    
    hover_text = [
        f"Media: {G.nodes[n]['media']}<br>Polarity: {G.nodes[n]['polarity']}<br>Rating: {G.nodes[n]['rating']:.1f}<br>Intensity: {G.nodes[n]['intensity']:.1f}<br>Valence: {G.nodes[n]['valence']:.2f}"
        for n in G.nodes
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(128,128,128,0.06)', width=0.5),
        hoverinfo='none', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=9, color=node_colors, opacity=0.9, line=dict(width=0)),
        text=hover_text, hoverinfo='text', showlegend=False
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='black', aspectmode='cube'
        ),
        paper_bgcolor='black', plot_bgcolor='black',
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(bgcolor='black', font=dict(color='white', size=11)),
        showlegend=False
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=2.0, y=2.0, z=1.0), center=dict(x=0, y=0, z=0)))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Stats")
    stats = st.session_state.rag.get_statistics()
    st.metric("KG Nodes", stats['total_nodes'])
    st.metric("LLM", "Off" if not stats['llm_available'] else "On")
    st.caption("Red = Detrimental · Blue = Beneficial")

# Chat / RAG Query Interface
st.subheader("🔍 Query the Knowledge Graph")
prompt = st.chat_input("Ask about chills patterns, polarity, or stimuli...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching KG..."):
            results = st.session_state.rag.query(prompt, n_results=5)
            if results:
                for i, r in enumerate(results):
                    with st.expander(f"Result {i+1} — Media: {r['metadata']['media']} (Score: {r['similarity_score']:.3f})"):
                        st.write(f"**Polarity:** {r['metadata']['polarity']}")
                        st.write(f"**Rating:** {r['metadata']['rating']:.1f}")
                        st.write(f"**Intensity:** {r['metadata']['intensity']:.1f}")
                        st.write(f"**Valence:** {r['metadata']['valence']:.2f}")
                        st.write(f"**Modality:** {r['metadata'].get('modality', 'Unknown')}")
                        st.write(f"**Context:** {r['metadata'].get('context', 'N/A')}/4")
                        st.caption(f"Distance: {r['distance']:.4f}")
            else:
                st.write("No matching nodes found.")
    
    st.session_state.messages.append({"role": "assistant", "content": f"Found {len(results)} results" if results else "No results"})
