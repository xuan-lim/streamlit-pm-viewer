import streamlit as st
import pandas as pd

# Optional: Configure the page layout to be wide
st.set_page_config(layout="wide")

st.title("📊 Project Management Excel Viewer")

st.markdown("""
Upload your project management Excel file (.xlsx or .xls) to view its details.
This tool allows you to quickly inspect the content of your project spreadsheets.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read the Excel file into a pandas DataFrame
        # header=0 assumes the first row is the header, which is common for PM excels
        df = pd.read_excel(uploaded_file, engine='openpyxl')

        st.success("File successfully uploaded and processed!")
        st.subheader("📋 Project Data Overview")

        # Display the DataFrame interactively
        # use_container_width makes the table stretch to fill the column
        st.dataframe(df, use_container_width=True)

        # --- Optional: Add more specific insights or visualizations ---
        st.markdown("---") # A separator line
        st.subheader("💡 Quick Insights")

        # Example 1: Check for a 'Status' column and display unique statuses
        if 'Status' in df.columns:
            st.write("#### Tasks by Status:")
            status_counts = df['Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            st.dataframe(status_counts, use_container_width=True)
            st.bar_chart(status_counts.set_index('Status'))
        else:
            st.info("No 'Status' column found for status analysis.")

        # Example 2: Display basic info about the DataFrame (columns, non-null counts, dtypes)
        st.write("#### DataFrame Information:")
        # df.info() prints to console, so we capture it to display in Streamlit
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        # Example 3: Display basic descriptive statistics for numerical columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            st.write("#### Numerical Column Statistics:")
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        else:
            st.info("No numerical columns found for statistical analysis.")


    except Exception as e:
        # Catch any errors during file reading or processing
        st.error(f"❌ Error processing your file: {e}")
        st.info("Please ensure it's a well-formatted Excel file (.xlsx or .xls) and not corrupted.")
        st.warning("Common issues include incorrect sheet names (if specified), or malformed data.")

st.markdown("---")
st.caption("✨ Built with Streamlit | For efficient project data viewing.")
