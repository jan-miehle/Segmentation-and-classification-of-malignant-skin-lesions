import streamlit as st
import sys
sys.path.append("..")
import state
"""
displays page for intensitiy augmentation
"""
def show():
    st.title("Intensity Augmentation")
    col1i, col2i, col3i = st.columns(3)
    with col1i:
        state.gauss_noise = st.checkbox('Gaussian noise', value = state.gauss_noise )
        if state.gauss_noise:
            state.gaus_prop = st.slider('Propability',0.0, 1.0, state.gaus_prop, key="gaus_prop")
            #RandFlip(prob = 0.5),
    with col2i:
        state.gibbs_noise = st.checkbox('Gibbs noise', value=state.gibbs_noise)
        if state.gibbs_noise:
            state.p_gibs = st.slider('Propability',0.0, 1.0, state.p_gibs, key="gibbs_prop")
    with col3i:
        if state.gibbs_noise and state.gauss_noise:
            state.one_off = st.checkbox('Use OneOff for Gibs- and Gaussnoise?', value = state.one_off )


    state.RandHistogramShift = st.checkbox('Random histogram shift', value=state.RandHistogramShift)
    if state.RandHistogramShift:
        col1y, col2y =st.columns(2)
        with col1y:
            state.hist_prop = st.slider('Propability',0.0, 1.0, state.hist_prop, key="hist_prop_prop")
        with col2y:
            state.control_points = st.slider('Control points',1, 30, state.control_points, key="control_points")   



    state.RandCoarseDropout = st.checkbox('RandCoarseDropout', value=state.RandCoarseDropout )
    if state.RandCoarseDropout:
        col1z, col2z, col3z =st.columns(3)
        with col1z:
            state.coarse_prop = st.slider('Propability',0.0, 1.0, state.coarse_prop, key="coarse_prop")
        with col2z:
            state.holes = st.slider('Number of holes',1, 100, state.holes, key="holes")
        with col3z:
            state.hole_size = st.slider('Hole-size',1, 100, state.hole_size, key="hole_size")



      