

import streamlit as st
import sys
sys.path.append("..")
import state

"""
displays page for network configuration
"""
def show():
       
        st.header("Model config")
        col1b, col2b, col3b = st.columns(3)
        with col1b:
            state.dropout = st.slider('Select a dropout-rate',0.0, 1.0, state.dropout)
        with col2b:
            state.smallest_layer = st.select_slider('Select first feature layer', options=[4, 8, 16, 32, 64, 128], value=state.smallest_layer)
        with col3b:
            if  state.largest_layer  <= state.smallest_layer or state.largest_layer > state.smallest_layer*32:
                state.largest_layer = state.smallest_layer  * 2
            state.largest_layer = st.select_slider('Select last feature layer', options=[state.smallest_layer*2,state.smallest_layer*4, state.smallest_layer*8, state.smallest_layer*16, state.smallest_layer*32], value=state.largest_layer)
        st.text("")

