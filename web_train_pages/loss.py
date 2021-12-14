import streamlit as st
import sys
sys.path.append("..")
import state
"""
displays page for selection of loss funtion
"""
def show():
        st.header("Loss Function")
        col1, col2, col3 = st.columns(3)
        with col1:
                if state.loss_name == 'Dice Loss':
                    i=0
                elif state.loss_name == 'Dice CE Loss':
                    i = 1
                else:
                    i = 2
                state.loss_name = st.selectbox('Select a loss function',('Dice Loss', 'Dice CE Loss', 'Dice Focal Loss'), index=i)
        with col2:
            st.text("")
            st.text("")
            state.jac = st.checkbox('Use Jaccard Index?', value=state.jac)
        with col3:
            st.text("")
            st.text("")
            state.sq = st.checkbox('Squared prediction?', value=state.sq)
        st.text("")

