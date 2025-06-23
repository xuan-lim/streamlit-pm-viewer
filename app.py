import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic PM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dynamic Project Management Dashboard")

st.markdown("""
Upload your project management Excel file (.xlsx or .xls) to visualize your project data.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        # --- Data Cleaning and Preprocessing ---
        # Convert date columns to datetime objects
        date_cols = ['Start Date', 'End Date', 'Delivered Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Fill NaN in 'Parent ID' for top-level projects (assuming they have no parent)
        if 'Parent ID' in df.columns:
            df['Parent ID'] = df['Parent ID'].fillna('')
        else:
            df['Parent ID'] = ''  # Ensure 'Parent ID' column exists even if missing

        # Create a unique ID for each item if not present, or ensure it's string
        if 'ID' not in df.columns:
            df['ID'] = [f"Item_{i}" for i in range(len(df))]
        df['ID'] = df['ID'].astype(str)

        # Ensure 'Name' and 'Type' columns exist and are string
        for col in ['Name', 'Type']:
            if col not in df.columns:
                st.error(f"Missing required column: `{col}`. Please check your Excel file. Essential columns: ID, Name, Type, Start Date, End Date.")
                st.stop()
            df[col] = df[col].astype(str).fillna('')

        # Handle 'Progress' column: Use existing or infer
        if 'Progress' not in df.columns:
            st.info("No 'Progress' column found. Inferring progress: 100% if delivered, 0% otherwise. Please add a 'Progress' column (0-100) for more accuracy.")
            df['Progress'] = df['Delivered Date'].apply(lambda x: 100 if pd.notna(x) else 0)
        else:
            df['Progress'] = pd.to_numeric(df['Progress'], errors='coerce').fillna(0) # Ensure numeric, fill NaNs with 0

        # Create a 'Status' column if missing, or standardize it
        if 'Status' not in df.columns:
            st.info("No 'Status' column found. Inferring status based on dates. Please add a 'Status' column (e.g., 'In Progress', 'Completed') for better control.")
            df['Status'] = 'N/A' # Default
        else:
            df['Status'] = df['Status'].astype(str).str.lower().fillna('n/a')

        # Infer status for better alerts if status column is generic or missing
        today = pd.to_datetime(datetime.now().date())
        def infer_status(row):
            if pd.notna(row['Delivered Date']) and row['Delivered Date'] <= row['End Date']:
                return 'completed'
            elif pd.notna(row['End Date']) and row['End Date'] < today:
                return 'overdue'
            elif pd.notna(row['Start Date']) and pd.notna(row['End Date']) and row['Start Date'] <= today and row['End Date'] >= today and row.get('Status', '').lower() not in ['completed', 'overdue']:
                return 'in progress'
            elif row.get('Status', '').lower() == 'n/a' or pd.isna(row.get('Status')): # Only overwrite generic 'N/A' status
                return 'not started'
            return row['Status'] # Keep existing status if it's specific

        df['Status'] = df.apply(infer_status, axis=1)


        st.success("Excel file successfully uploaded and processed!")

        # Store DataFrame in session state to avoid re-uploading
        st.session_state.df = df

        # --- TEMPORARY: Display DataFrame head to verify processing ---
        st.subheader("Preview of Processed Data")
        st.dataframe(df.head())
        # --- END TEMPORARY ---

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.info("Please ensure it's a valid Excel file (.xlsx or .xls) and that required columns (ID, Name, Type, Start Date, End Date) are present and correctly formatted.")
        st.stop()

# --- Keep the rest of the main app logic COMMENTED OUT for now ---
# if 'df' in st.session_state:
#    df = st.session_state.df
#    st.sidebar.header("Filter Projects")
#    # ... all other code
