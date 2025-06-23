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

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.info("Please ensure it's a valid Excel file (.xlsx or .xls) and that required columns (ID, Name, Type, Start Date, End Date) are present and correctly formatted.")
        st.stop()

# --- Main App Logic (only if DataFrame is loaded) ---
if 'df' in st.session_state:
    df = st.session_state.df

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Projects")
    all_types = df['Type'].unique().tolist()
    selected_types = st.sidebar.multiselect("Filter by Type", all_types, default=all_types)

    all_statuses = df['Status'].unique().tolist()
    selected_statuses = st.sidebar.multiselect("Filter by Status", all_statuses, default=all_statuses)

    # Apply filters
    filtered_df = df[
        df['Type'].isin(selected_types) &
        df['Status'].isin(selected_statuses)
    ].copy() # Ensure a copy to avoid SettingWithCopyWarning

    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    selected_tab = st.sidebar.radio(
        "Go to:",
        ["Dashboard Overview", "Full Data Table"]
    )

    if selected_tab == "Dashboard Overview":
        # --- Dashboard Layout ---
        col1, col2 = st.columns([0.7, 0.3]) # Adjust ratios as needed

        with col1: # Left/Main Column for Timeline and Hierarchy
            st.header("Project Timeline (Gantt Chart)")
            st.markdown("Interact with the chart (zoom, pan) using the toolbar. Use sliders below to adjust date range.")

            # Date Range Slider for "Zoom"
            min_date = filtered_df['Start Date'].min() if not filtered_df['Start Date'].empty else datetime.now() - timedelta(days=30)
            max_date = filtered_df['End Date'].max() if not filtered_df['End Date'].empty else datetime.now() + timedelta(days=365)

            if pd.isna(min_date): min_date = datetime.now() - timedelta(days=30)
            if pd.isna(max_date): max_date = datetime.now() + timedelta(days=365)

            # Ensure min_date and max_date are actual datetime objects for the slider
            min_date = pd.to_datetime(min_date.date())
            max_date = pd.to_datetime(max_date.date())

            date_range_start, date_range_end = st.slider(
                "Select Date Range for Timeline:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )

            # Filter data for Gantt chart based on selected date range
            gantt_df = filtered_df[
                (filtered_df['Start Date'] <= date_range_end) &
                (filtered_df['End Date'] >= date_range_start)
            ].sort_values(by='Start Date')

            if not gantt_df.empty:
                # Add a text column for hover details including progress
                gantt_df['Text'] = gantt_df.apply(
                    lambda row: f"{row['Name']} ({row['Type']})<br>Status: {row['Status'].capitalize()}<br>Progress: {row['Progress']:.0f}%", axis=1
                )
                # Use 'Type' for color, or a custom color map
                fig = px.timeline(
                    gantt_df,
                    x_start="Start Date",
                    x_end="End Date",
                    y="Name",
                    color="Type", # You can change this to 'Status' for example
                    text="Progress", # Display progress on bars if space allows
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
                fig.update_yaxes(autorange="reversed") # Projects on top
                fig.update_layout(height=600, showlegend=True) # Adjust height as needed
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No projects to display in the selected date range or with current filters.")

            st.header("📋 Hierarchical Project View")
            st.markdown("Expand projects to see their sub-projects, tasks, and milestones.")

            # Function to build and display hierarchy (now uses filtered_df)
            def display_children(parent_id, level=0):
                children = filtered_df[filtered_df['Parent ID'] == parent_id]
                if children.empty:
                    return

                for index, row in children.iterrows():
                    prefix = "  " * level
                    # Removed 'key' for expander for compatibility
                    
                    progress_bar = f" {int(row['Progress'])}%"
                    
                    if row['Type'] in ['Project', 'Sub-Project'] and row['ID'] in filtered_df['Parent ID'].values:
                        with st.expander(f"{prefix}📁 **{row['Name']}** ({row['Type']}) {progress_bar} - Status: {row.get('Status', 'N/A').title()}"):
                            st.write(f"- **ID:** {row['ID']}")
                            st.write(f"- **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                            st.write(f"- **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                            st.write(f"- **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                            st.progress(int(row['Progress']) / 100) # Progress bar
                            display_children(row['ID'], level + 1)
                    else: # It's a Task, Milestone, or a Project/Sub-Project without further children
                        status_emoji = "✅" if row['Status'] == 'completed' else "⏳" if row['Status'] == 'in progress' else "🔴" if row['Status'] == 'overdue' else "⚪"
                        st.markdown(f"{prefix}{status_emoji} **{row['Name']}** ({row['Type']}) {progress_bar}")
                        st.markdown(f"{prefix}  - **ID:** {row['ID']}")
                        st.markdown(f"{prefix}  - **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                        st.markdown(f"{prefix}  - **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                        st.markdown(f"{prefix}  - **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                        st.markdown(f"{prefix}  - **Status:** {row.get('Status', 'N/A').title()}")
                        st.markdown(f"{prefix}  - **Assigned To:** {row.get('Assigned To', 'N/A')}")
                        st.progress(int(row['Progress']) / 100) # Progress bar for leaf nodes

            # Get top-level items (no parent, and must be in filtered_df)
            top_level_projects = filtered_df[filtered_df['Parent ID'].isna() | (filtered_df['Parent ID'] == '')]

            if not top_level_projects.empty:
                for index, row in top_level_projects.iterrows():
                    progress_bar = f" {int(row['Progress'])}%"
                    if row['Type'] in ['Project', 'Program'] or row['ID'] in filtered_df['Parent ID'].values:
                        # Removed 'key' for expander for compatibility
                        with st.expander(f"📌 **{row['Name']}** ({row['Type']}) {progress_bar} - Status: {row.get('Status', 'N/A').title()}", expanded=True):
                            st.write(f"- **ID:** {row['ID']}")
                            st.write(f"- **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                            st.write(f"- **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                            st.write(f"- **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                            st.progress(int(row['Progress']) / 100) # Progress bar
                            display_children(row['ID'], 1)
                    else: # A top-level item that doesn't act as a parent (e.g., a standalone task)
                        status_emoji = "✅" if row['Status'] == 'completed' else "⏳" if row['Status'] == 'in progress' else "🔴" if row['Status'] == 'overdue' else "⚪"
                        st.markdown(f"**{status_emoji} {row['Name']}** ({row['Type']}) {progress_bar} - Status: {row.get('Status', 'N/A').title()}")
                        st.markdown(f"  - **ID:** {row['ID']}")
                        st.markdown(f"  - **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                        st.markdown(f"  - **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                        st.markdown(f"  - **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                        st.markdown(f"  - **Assigned To:** {row.get('Assigned To', 'N/A')}")
                        st.progress(int(row['Progress']) / 100) # Progress bar

            else:
                st.info("No top-level projects found or all filtered out. Ensure 'Parent ID' is empty for your main projects.")


        with col2: # Right Column for Alerts and Summaries
            st.header("🔔 Alerts")
            today = pd.to_datetime(datetime.now().date())

            # Filter active items for alerts (not completed or overdue due to delayed delivery)
            alert_items = filtered_df[
                ((filtered_df['Status'].isin(['in progress', 'not started', 'on hold', 'overdue'])) |
                 (filtered_df['Delivered Date'].notna() & (filtered_df['Delivered Date'] > filtered_df['End Date']))) &
                (filtered_df['End Date'].notna()) # Ensure End Date exists for comparison
            ].copy()

            st.subheader("❗ Overdue Items")
            overdue_items = alert_items[alert_items['End Date'] < today].sort_values(by='End Date')
            if not overdue_items.empty:
                for idx, item in overdue_items.head(5).iterrows(): # Show top 5 overdue
                    st.error(f"**{item['Name']}** ({item['Type']}) is overdue! (Due: {item['End Date'].strftime('%Y-%m-%d')})")
            else:
                st.info("🎉 No overdue items currently!")

            st.subheader("🔜 Due Soon (Next 7 Days)")
            due_soon_items = alert_items[
                (alert_items['End Date'] >= today) &
                (alert_items['End Date'] <= today + timedelta(days=7))
            ].sort_values(by='End Date')

            if not due_soon_items.empty:
                for idx, item in due_soon_items.head(5).iterrows(): # Show top 5 due soon
                    st.warning(f"**{item['Name']}** ({item['Type']}) due in {(item['End Date'] - today).days} days! (End: {item['End Date'].strftime('%Y-%m-%d')})")
            else:
                st.info("Nothing due in the next 7 days.")

            st.subheader("✅ Recently Completed (Last 30 Days)")
            completed_recently = filtered_df[
                (filtered_df['Delivered Date'].notna()) &
                (filtered_df['Delivered Date'] >= today - timedelta(days=30))
            ].sort_values(by='Delivered Date', ascending=False)
            if not completed_recently.empty:
                for idx, item in completed_recently.head(5).iterrows(): # Show top 5 recently completed
                    st.success(f"**{item['Name']}** ({item['Type']}) completed! (Delivered: {item['Delivered Date'].strftime('%Y-%m-%d')})")
            else:
                st.info("No items completed recently.")


            st.header("📈 Project Overview Summary")
            
            # KPI Metrics
            total_projects = filtered_df[filtered_df['Type'] == 'Project'].shape[0]
            active_projects = filtered_df[
                (filtered_df['Type'] == 'Project') &
                (filtered_df['Status'].isin(['in progress', 'not started', 'on hold']))
            ].shape[0]
            completed_projects = filtered_df[
                (filtered_df['Type'] == 'Project') &
                (filtered_df['Status'] == 'completed')
            ].shape[0]

            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            kpi_col1.metric("Total Projects", total_projects)
            kpi_col2.metric("Active Projects", active_projects)
            kpi_col3.metric("Completed Projects", completed_projects)

            summary_granularity = st.selectbox(
                "Summarize by Timeframe:",
                ["Weekly", "Monthly", "Quarterly", "Half-Yearly", "Yearly"],
                index=0 # Default to Weekly
            )

            st.markdown(f"**Summary of items by {summary_granularity} (based on End Date):**")

            summary_df_filtered = filtered_df.dropna(subset=['End Date']).copy()


            def get_time_period(date, granularity):
                if granularity == "Daily": # Not in selectbox, but good for internal consistency
                    return date.strftime('%Y-%m-%d')
                elif granularity == "Weekly":
                    return f"{date.isocalendar()[0]}-W{date.isocalendar()[1]:02d}"
                elif granularity == "Monthly":
                    return date.strftime('%Y-%m')
                elif granularity == "Quarterly":
                    quarter = (date.month - 1) // 3 + 1
                    return f"{date.year}-Q{quarter}"
                elif granularity == "Half-Yearly":
                    half_year = 1 if date.month <= 6 else 2
                    return f"{date.year}-H{half_year}"
                elif granularity == "Yearly":
                    return str(date.year)
                return "N/A"

            if not summary_df_filtered.empty:
                summary_df_filtered['Time Period'] = summary_df_filtered['End Date'].apply(lambda x: get_time_period(x, summary_granularity))

                grouped_summary = summary_df_filtered.groupby('Time Period').agg(
                    Total_Items=('ID', 'count'),
                    Completed_Items=('Delivered Date', lambda x: x.notna().sum())
                ).reset_index()

                grouped_summary['Overdue_Items'] = summary_df_filtered[summary_df_filtered['Status'] == 'overdue'].groupby('Time Period')['ID'].count().reindex(grouped_summary['Time Period'], fill_value=0).values
                grouped_summary['In_Progress_Items'] = summary_df_filtered[summary_df_filtered['Status'] == 'in progress'].groupby('Time Period')['ID'].count().reindex(grouped_summary['Time Period'], fill_value=0).values
                grouped_summary['Not_Started_Items'] = summary_df_filtered[summary_df_filtered['Status'] == 'not started'].groupby('Time Period')['ID'].count().reindex(grouped_summary['Time Period'], fill_value=0).values

                grouped_summary = grouped_summary.sort_values(by='Time Period')

                st.dataframe(grouped_summary, use_container_width=True)

                st.subheader("Items by Status over Time")
                # Prepare data for stacked bar chart
                chart_data_melted = grouped_summary.melt(
                    id_vars='Time Period',
                    value_vars=['Completed_Items', 'In_Progress_Items', 'Not_Started_Items', 'Overdue_Items'],
                    var_name='Status Type',
                    value_name='Count'
                )
                fig_summary = px.bar(
                    chart_data_melted,
                    x='Time Period',
                    y='Count',
                    color='Status Type',
                    title=f'Item Status by {summary_granularity}',
                    barmode='stack'
                )
                st.plotly_chart(fig_summary, use_container_width=True)

            else:
                st.info("No data available for the selected time summary.")

    elif selected_tab == "Full Data Table":
        st.header("Raw Project Data")
        st.markdown("This table shows all data from your uploaded Excel file after initial processing.")
        st.dataframe(filtered_df, use_container_width=True)


    st.markdown("---")
    st.caption(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Current time zone).")
