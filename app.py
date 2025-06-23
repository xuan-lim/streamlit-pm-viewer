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
        date_cols = ['Start Date', 'End Date', 'Delivered Date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'Parent ID' in df.columns:
            df['Parent ID'] = df['Parent ID'].fillna('')
        else:
            df['Parent ID'] = ''
        df['ID'] = [f"Item_{i}" for i in range(len(df))] if 'ID' not in df.columns else df['ID'].astype(str)
        for col in ['Name', 'Type']:
            if col not in df.columns:
                st.error(f"Missing required column: `{col}`. Please check your Excel file. Essential columns: ID, Name, Type, Start Date, End Date.")
                st.stop()
            df[col] = df[col].astype(str).fillna('')

        if 'Progress' not in df.columns:
            st.info("No 'Progress' column found. Inferring progress: 100% if delivered, 0% otherwise. Please add a 'Progress' column (0-100) for more accuracy.")
            df['Progress'] = df['Delivered Date'].apply(lambda x: 100 if pd.notna(x) else 0)
        else:
            df['Progress'] = pd.to_numeric(df['Progress'], errors='coerce').fillna(0)

        if 'Status' not in df.columns:
            st.info("No 'Status' column found. Inferring status based on dates. Please add a 'Status' column (e.g., 'In Progress', 'Completed') for better control.")
            df['Status'] = 'N/A'
        else:
            df['Status'] = df['Status'].astype(str).lower().fillna('n/a')

        today = pd.to_datetime(datetime.now().date())
        def infer_status(row):
            if pd.notna(row['Delivered Date']) and row['Delivered Date'] <= row['End Date']:
                return 'completed'
            elif pd.notna(row['End Date']) and row['End Date'] < today:
                return 'overdue'
            elif pd.notna(row['Start Date']) and pd.notna(row['End Date']) and row['Start Date'] <= today and row['End Date'] >= today and row.get('Status', '').lower() not in ['completed', 'overdue']:
                return 'in progress'
            elif row.get('Status', '').lower() == 'n/a' or pd.isna(row.get('Status')):
                return 'not started'
            return row['Status']

        df['Status'] = df.apply(infer_status, axis=1)

        st.success("Excel file successfully uploaded and processed!")

        st.session_state.df = df

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.info("Please ensure it's a valid Excel file (.xlsx or .xls) and that required columns (ID, Name, Type, Start Date, End Date) are present and correctly formatted.")
        st.stop()

# --- Start of Main App Logic - THIS BLOCK MUST NOT BE INDENTED ---
if 'df' in st.session_state: # This line must start at the very first character of the file.
    df = st.session_state.df

    st.sidebar.header("Filter Projects")
    all_types = df['Type'].unique().tolist()
    selected_types = st.sidebar.multiselect("Filter by Type", all_types, default=all_types)

    all_statuses = df['Status'].unique().tolist()
    selected_statuses = st.sidebar.multiselect("Filter by Status", all_statuses, default=all_statuses)

    # Apply filters
    filtered_df = df[
        df['Type'].isin(selected_types) &
        df['Status'].isin(selected_statuses)
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    selected_tab = st.sidebar.radio(
        "Go to:",
        ["Dashboard Overview", "Full Data Table"]
    )

    if selected_tab == "Dashboard Overview":
        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            st.header("Project Timeline (Gantt Chart)")
            st.markdown("Interact with the chart (zoom, pan) using the toolbar. Use sliders below to adjust date range.")

            min_date_df = filtered_df['Start Date'].min()
            max_date_df = filtered_df['End Date'].max()

            min_date = min_date_df.date() if pd.notna(min_date_df) else (datetime.now().date() - timedelta(days=90))
            max_date = max_date_df.date() if pd.notna(max_date_df) else (datetime.now().date() + timedelta(days=180))

            if min_date > max_date:
                min_date = max_date - timedelta(days=7)

            today_date = datetime.now().date()
            default_slider_start = max(min_date, today_date - timedelta(days=30))
            default_slider_end = min(max_date, today_date + timedelta(days=90))

            if default_slider_start > default_slider_end:
                 default_slider_start = default_slider_end - timedelta(days=7)

            date_range_start, date_range_end = st.slider(
                "Select Date Range for Timeline:",
                min_value=min_date,
                max_value=max_date,
                value=(default_slider_start, default_slider_end),
                format="YYYY-MM-DD"
            )

            gantt_df = filtered_df[
                (filtered_df['Start Date'] <= pd.to_datetime(date_range_end)) &
                (filtered_df['End Date'] >= pd.to_datetime(date_range_start))
            ].sort_values(by='Start Date')

            if not gantt_df.empty:
                gantt_df['Text'] = gantt_df.apply(
                    lambda row: f"{row['Name']} ({row['Type']})<br>Status: {row['Status'].capitalize()}<br>Progress: {row['Progress']:.0f}%", axis=1
                )
                fig = px.timeline(
                    gantt_df,
                    x_start="Start Date",
                    x_end="End Date",
                    y="Name",
                    color="Type",
                    text="Progress",
                    title="Project Timeline Overview",
                    hover_name="Name",
                    hover_data={
                        "Type": True,
                        "Status": True,
                        "Start Date": "|%Y-%m-%d",
                        "End Date": "|%Y-%m-%d",
                        "Delivered Date": "|%Y-%m-%d",
                        "Progress": ":.0f%",
                        "Parent ID": True
                    }
                )
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No projects to display in the selected date range or with current filters.")

            # --- Hierarchical Project View (still commented out for now) ---
            # st.header("📋 Hierarchical Project View")
            # ... (all display_children and top_level_projects logic) ...

        with col2: # This is the end of the 'with' statement for col2 (Line 172 in your context)
            # --- Alerts (still commented out for now) ---
            # st.header("🔔 Alerts")
            # ... (all alert logic) ...

            # --- Project Overview Summary (still commented out for now) ---
            # st.header("📈 Project Overview Summary")
            # ... (all summary logic) ...

    # This 'elif' (Line 181 in your context) MUST be aligned with the 'if' on line 138 (if selected_tab == "Dashboard Overview":)
    elif selected_tab == "Full Data Table":
        st.header("Raw Project Data")
        st.markdown("This table shows all data from your uploaded Excel file after initial processing.")
        st.dataframe(filtered_df, use_container_width=True)

    st.markdown("---")
    st.caption(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Current time zone).")
