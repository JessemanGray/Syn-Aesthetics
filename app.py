# app.py — SynAesthetics Dashboard
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import json
from pathlib import Path
from data_processor import SynAestheticsDataProcessor
from chills_rag import ChillsRAGSystem

st.set_page_config(page_title="SynAesthetics", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    * {
        font-size: 14px !important;
        line-height: 1.3 !important;
    }
    .title {
        font-family: monospace;
        font-size: 22px !important;
        font-weight: 400;
        text-transform: uppercase;
        color: white;
        margin-bottom: 4px;
    }
    .stimuli-list {
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 4px;
        padding: 4px 6px;
    }
    .stimuli-item {
        padding: 2px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 13px !important;
    }
    .stimuli-item a {
        color: white;
        text-decoration: none;
        font-size: 13px !important;
    }
    .stimuli-item a:hover {
        color: #7ab7ff;
        text-decoration: underline;
    }
    .stimuli-meta {
        color: #666;
        font-size: 11px !important;
        margin-left: 4px;
    }
    .stDataFrame {
        font-size: 12px !important;
    }
    .stDataFrame th, .stDataFrame td {
        font-size: 12px !important;
        padding: 2px 4px !important;
    }
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .stColumns {
        gap: 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

if "processor" not in st.session_state:
    with st.spinner("Loading..."):
        st.session_state.processor = SynAestheticsDataProcessor()
        df, G = st.session_state.processor.run_pipeline()
        st.session_state.G = G
        st.session_state.rag = ChillsRAGSystem(graph=G, use_llm=True)
        st.session_state.df = df
        st.session_state.selected_node = None
        if Path("model_metrics.json").exists():
            with open("model_metrics.json", "r") as f:
                st.session_state.metrics = json.load(f)
        else:
            st.session_state.metrics = {}

st.markdown('<div class="title">SynAesthetics</div>', unsafe_allow_html=True)

graph_col, panel_col = st.columns([4.2, 1.2], gap="small")

with graph_col:
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
    
    def get_node_color(node):
        if 'chills_ratio' in G.nodes[node] and G.nodes[node]['chills_ratio'] > 0:
            return G.nodes[node]['chills_ratio']
        elif 'intensity' in G.nodes[node]:
            return G.nodes[node]['intensity'] / 100
        else:
            return 0.5
    node_colors = [get_node_color(n) for n in G.nodes]
    
    hover_text = []
    for n in G.nodes:
        node_data = G.nodes[n]
        if node_data.get('type') == 'participant' or 'response' in node_data:
            hover_text.append(
                f"<b>{node_data.get('media', 'Unknown')}</b><br>"
                f"Polarity: {node_data.get('polarity', 'N/A')}<br>"
                f"Intensity: {node_data.get('intensity', 0):.1f}<br>"
                f"Valence: {node_data.get('valence', 0):.2f}<br>"
                f"Arousal: {node_data.get('arousal', 0):.2f}<br>"
                f"Context: {node_data.get('context', 0)}/4"
            )
        else:
            modality = node_data.get('modality', 'Unknown')
            date_str = node_data.get('date', '')
            if not date_str and 'birth' in node_data:
                date_str = node_data['birth']
            hover_text.append(
                f"<b>{node_data.get('media', 'Unknown')}</b><br>"
                f"Modality: {modality}<br>"
                f"Date: {date_str if date_str else 'N/A'}"
            )
    
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
        marker=dict(
            size=10,
            color=node_colors,
            colorscale='Viridis',
            opacity=0.5,
            line=dict(width=0)
        ),
        hovertext=hover_text,
        hoverinfo='text',
        showlegend=False
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='black', aspectmode='cube'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        margin=dict(l=0, r=0, b=0, t=0),
        hoverlabel=dict(bgcolor='black', font=dict(color='white', size=11)),
        showlegend=False,
        height=600
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=1.2, y=1.2, z=0.8), center=dict(x=0, y=0, z=0)))
    st.plotly_chart(fig, use_container_width=True)

with panel_col:
    if hasattr(st.session_state.processor, 'df_stimuli') and st.session_state.processor.df_stimuli is not None:
        stim_df = st.session_state.processor.df_stimuli
        st.markdown('<div class="stimuli-list">', unsafe_allow_html=True)
        for _, row in stim_df.iterrows():
            name = row['Stimulus']
            modality = row.get('Modality', '')
            date_str = row.get('Date', '')
            if not date_str:
                date_str = row.get('Year', '')
            if st.session_state.df is not None:
                stim_responses = st.session_state.df[st.session_state.df['Media_Label'] == name]
                total = len(stim_responses)
                pos = len(stim_responses[stim_responses['Polarity'] == 'Beneficial'])
                neg = len(stim_responses[stim_responses['Polarity'] == 'Detrimental'])
                avg_int = stim_responses['Intensity'].mean() if not stim_responses.empty else 0
            else:
                total = pos = neg = 0
                avg_int = 0
            stim_id = row.get('Stimulus ID', '')
            link = f"https://chillstv.com/media/{stim_id}" if stim_id else f"https://chillstv.com/#stimulus={name.replace(' ', '%20')}"
            display = f"{name.upper()} ({modality}, {date_str})" if modality and date_str else f"{name.upper()} ({modality})" if modality else f"{name.upper()} ({date_str})" if date_str else name.upper()
            st.markdown(
                f'<div class="stimuli-item">'
                f'<a href="{link}" target="_blank">{display}</a>'
                f'<span class="stimuli-meta">{total} resp, +{pos}/-{neg}, {avg_int:.1f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)