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
        st.session_state.rag = ChillsRAGSystem(graph=G, use_llm=True)
        st.session_state.messages = []
        st.session_state.df = df

# --- HEADER ---
st.title("🧊 ChillsDB Knowledge Graph Explorer")
st.caption("Interactive visualization of aesthetic chills responses with RAG-powered querying")

# --- TOP METRICS ROW ---
col1, col2, col3, col4, col5 = st.columns(5)
stats = st.session_state.rag.get_statistics()
df_processed = st.session_state.df

if df_processed is not None:
    polarity_counts = df_processed['Polarity'].value_counts()
    beneficial = polarity_counts.get('Beneficial', 0) + polarity_counts.get('Mildly Beneficial', 0)
    detrimental = polarity_counts.get('Detrimental', 0) + polarity_counts.get('Mildly Detrimental', 0)
else:
    beneficial = 0
    detrimental = 0

col1.metric("KG Nodes", stats['total_nodes'])
col2.metric("Beneficial", beneficial, delta="+")
col3.metric("Detrimental", detrimental, delta="-")
col4.metric("Avg Intensity", f"{df_processed['Intensity'].mean():.1f}" if df_processed is not None else "N/A")
col5.metric("Model", "Pleias-RAG-1B" if stats['llm_available'] else "Off")

# --- MAIN LAYOUT: Graph + Sidebar ---
left_col, right_col = st.columns([4, 1])

with left_col:
    st.subheader("🌐 3D Knowledge Graph")
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
        f"Media: {G.nodes[n]['media']}<br>"
        f"Polarity: {G.nodes[n]['polarity']}<br>"
        f"Rating: {G.nodes[n]['rating']:.1f}<br>"
        f"Intensity: {G.nodes[n]['intensity']:.1f}<br>"
        f"Valence: {G.nodes[n]['valence']:.2f}<br>"
        f"Chills Ratio: {G.nodes[n].get('chills_ratio', 0):.2f}<br>"
        f"Blob Polarity: {G.nodes[n].get('blob_polarity', 'neutral')}"
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
        marker=dict(size=12, color=node_colors, opacity=0.9, line=dict(width=0)),
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
        showlegend=False,
        height=700
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=2.0, y=2.0, z=1.0), center=dict(x=0, y=0, z=0)))
    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.subheader("📊 Quick Stats")
    if df_processed is not None:
        st.markdown("**Polarity Distribution**")
        st.dataframe(polarity_counts.reset_index().rename(columns={'index': 'Polarity', 'Polarity': 'Count'}), height=150)
        st.markdown("**Intensity Clusters**")
        cluster_counts = df_processed['Intensity_Cluster'].value_counts()
        st.dataframe(cluster_counts.reset_index().rename(columns={'index': 'Cluster', 'Intensity_Cluster': 'Count'}), height=150)
        st.markdown("**Response Categories**")
        resp_counts = df_processed['Response_Category'].value_counts()
        st.dataframe(resp_counts.reset_index().rename(columns={'index': 'Category', 'Response_Category': 'Count'}), height=150)
    st.caption("🔴 Red = Detrimental · 🔵 Blue = Beneficial")

# --- CHAT / QUERY INTERFACE ---
st.subheader("🔍 Query the Knowledge Graph")
prompt = st.chat_input("Ask about chills patterns, polarity, intensity, or stimuli...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching KG and generating aligned response..."):
            result = st.session_state.rag.query(prompt, n_results=5, generate=True)
            
            if isinstance(result, str):
                # Generated response from the aesthetic alignment layer
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            elif isinstance(result, list) and result:
                # Fallback: raw retrieval results
                for i, r in enumerate(result):
                    with st.expander(f"Result {i+1} — Media: {r['metadata']['media']} (Score: {r['similarity_score']:.3f})"):
                        st.write(f"**Polarity:** {r['metadata']['polarity']}")
                        st.write(f"**Rating:** {r['metadata']['rating']:.1f}")
                        st.write(f"**Intensity:** {r['metadata']['intensity']:.1f}")
                        st.write(f"**Valence:** {r['metadata']['valence']:.2f}")
                        st.write(f"**Chills Ratio:** {r['metadata'].get('chills_ratio', 0):.2f}")
                        st.caption(f"Distance: {r['similarity_score']:.4f}")
                st.session_state.messages.append({"role": "assistant", "content": f"Found {len(result)} results"})
            else:
                st.write("No matching nodes found.")
                st.session_state.messages.append({"role": "assistant", "content": "No results"})
