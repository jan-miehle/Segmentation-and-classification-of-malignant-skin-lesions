import streamlit as st
import sys
sys.path.append("..")
import state
"""
displays page for spatial augmentation
"""
def show():
    st.header("Spatial image augmentation")
    state.flip = st.checkbox('Random flip', value = state.flip )
    if state.flip:
        s1, s2 = st.columns(2)
        with s1:
            state.spatial_axis= st.select_slider('spatial axis', options =[None, 0 , 1], key="spatialaxos", value=state.spatial_axis)
        with s2:
            state.p_flip = st.slider('propability',0.0, 1.0, state.p_flip, key="flipprov")
    state.rotate = st.checkbox('Random rotate', value=state.rotate)
    if state.rotate:
        s1a, s2a, s3a = st.columns(3)
        with s1a:
            state.range_x= st.slider('range x',0.0, 1.0, state.range_x, key="rangex")
        with s2a:
            state.range_y= st.slider('range y',0.0, 1.0, state.range_y, key="rangey")
        with s3a:
            state.p_rotate = st.slider('propability',0.0, 1.0, state.p_rotate, key="protate") 
    state.croping = st.checkbox('Random scale croping', value=state.croping)
    if state.croping:
        s1b, s2b = st.columns(2)
        with s1b:
            state.roi_scale = st.slider('min roi size',0.01, state.max_roi_scale - 0.01, state.roi_scale, key="roiscale") 
        with s2b:
            state.max_roi_scale = st.slider('max roi scale', state.roi_scale + 0.01, 0.99, state.max_roi_scale, key="maxroiscale")
    state.zoom = st.checkbox('random zoom', value=state.zoom )
    if state.zoom:
        s1c, s2c, s3c = st.columns(3)
        with s1c:
            state.min_zoom= st.slider('min zoom',0.0, 1.0, state.min_zoom, key="maxzoom")
        with s2c:
            state.max_zoom= st.slider('max zoom',state.min_zoom, 4.0, state.max_zoom, key="minzoom")
        with s3c:
            state.p_zoom= st.slider('propability',0.0, 1.0, state.p_zoom, key="pzoom") 
