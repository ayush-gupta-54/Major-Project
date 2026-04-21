import streamlit as st
import os
from parser import parse_file
from graph import build_graph
from model import GNN
import torch
import pyvis.network as net

st.set_page_config(layout="wide", page_title="Codebase Cartographer")

st.title("🌐 Semantic Codebase Cartography")
st.markdown("### Predicting Intent Dependencies via Multi-Relational GNNs")

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Python Source", type=['py'])

if uploaded_file:
    with open("temp_input.py", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing Code Semantics..."):
        parsed = parse_file("temp_input.py")
        data, node_index = build_graph(parsed)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Functions (Nodes)", len(parsed["functions"]))
        col2.metric("Calls (Edges)", len(parsed["calls"]))
        col3.metric("Embedding Dim", data.x.size(1))

        tab1, tab2 = st.tabs(["📍 3D Structural Map", "🧠 Semantic Clusters"])

        with tab1:
            g = net.Network(height='500px', width='100%', bgcolor='#222222', font_color='white', directed=True)
            for func in parsed["functions"]:
                g.add_node(func, label=func, color='#00ffcc')
            for src, dst in parsed["calls"]:
                g.add_edge(src, dst)

            g.save_graph("map.html")
            st.components.v1.html(open("map.html", 'r').read(), height=550)

        with tab2:
            st.write("Embedding visualization coming soon...")

    st.success("Analysis Complete!")