import streamlit as st
import sys
sys.path.append("..")
import state

"""
displays page for network input
"""
def show():
    st.header("Input settings")
    col1c, col2c, col3c = st.columns(3)
    with col1c:
        state.size = st.number_input('Select image size', min_value=32, value=state.size)
    with col2c:
        st.text("")
        st.text("")
        state.top_hat = st.checkbox('Use images edited with top hat transformation?', value=state.top_hat)
    
    with col3c:
        st.text("")
        st.text("")
        state.lbp = st.checkbox('Add LBP channel?', value=state.lbp) 


    #select image path according to selection above:    
    if state.lbp and state.top_hat:
        state.image_dir = "ph2_tophatv2t/4_channel_lbp"
        state.channels = 4

    elif state.lbp:
        state.image_dir = "image_+_lbp"
    elif state.top_hat:
        state.image_dir = "ph2_tophat_v2.3T" 
        state.channels = 3
    else:
        #if None, unet loads the standart ph2 folder
        state.iamge_dir = None
        state.channels = 3
    st.text("")