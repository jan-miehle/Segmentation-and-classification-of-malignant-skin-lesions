import streamlit as st
from torch.nn.functional import dropout

from PIL import Image

#subpages
import web_train_pages.loss
import web_train_pages.network
import web_train_pages.input
import web_train_pages.spatial
import web_train_pages.run
import web_train_pages.intensity


#main page for web interface


#built sidebar:
titles = ['Loss function', 'Network model', 'Input', 'Spatial augmentation', 'Intensity augmentation','Start']
p_list = [web_train_pages.loss,web_train_pages.network, web_train_pages.input, web_train_pages.spatial ,web_train_pages.intensity, web_train_pages.run]
p = st.sidebar.radio('',titles)
index = titles.index(p)
#show selected page:
p_list[index].show()


