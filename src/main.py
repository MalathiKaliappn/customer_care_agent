# import streamlit as st

# st.set_page_config(page_title="Customer Assistant", layout="wide")
# st.title("Customer Care Assistant")
# st.markdown("Use the uploader in the sidebar to upload the file manually and then ask the question using Chatbot")


import sys
import os
import streamlit as st

# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Now you can import the pages module
from pages import uploader, chatbot

# Set up Streamlit page config
st.set_page_config(page_title="Customer Care Assistant", layout="wide")

# Title and description
st.title("Customer Care Assistant")
st.markdown("Use the sidebar to navigate between uploading a file and chatting with the assistant.")

# Sidebar: Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["File Upload", "Chatbot"])

# Load the appropriate page
if page == "File Upload":
    uploader.render()

elif page == "Chatbot":
    chatbot.render()
