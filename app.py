import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px # Keep if you have it in requirements.txt

st.set_page_config(
    page_title="Test App",
    page_icon="📊",
    layout="wide"
)

st.title("Hello Streamlit!")
st.write("If you see this, the app is loading!")

# COMMENT OUT EVERYTHING BELOW THIS LINE FOR TESTING
# uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
# if uploaded_file is not None:
#     try:
#         df = pd.read_excel(uploaded_file, engine='openpyxl')
#         # ... and so on, comment out all your logic
