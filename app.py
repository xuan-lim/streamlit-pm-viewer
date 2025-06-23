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

        # Ensure ID and Parent ID are consistent strings and stripped of whitespace early
        df['ID'] = df['ID'].astype(str).str.strip()
        if 'Parent ID' in df.columns:
            df['Parent ID'] = df['Parent ID'].astype(str).str.strip().fillna('')
        else:
            df['Parent ID'] = '' # Ensure Parent ID column exists if not in original data

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
            df['Status'] = df['Status'].astype(str).str.lower().fillna('n/a')


        # Infer status for better alerts if status column is generic or missing/inconsistent
        today = pd.to_datetime(datetime.now().date())
        def infer_status(row):
            current_status = str(row.get('Status', 'n/a')).lower()

            if pd.notna(row['Delivered Date']) and pd.notna(row['End Date']) and row['Delivered Date'] <= row['End Date']:
                return 'completed'
            elif pd.notna(row['Delivered Date']) and pd.notna(row['End Date']) and row['Delivered Date'] > row['End Date']:
                 return 'completed (late)'
            elif pd.notna(row['End Date']) and row['End Date'] < today:
                return 'overdue'
            elif pd.notna(row['Start Date']) and pd.notna(row['End Date']) and row['Start Date'] <= today and row['End Date'] >= today:
                if current_status not in ['completed', 'overdue']:
                    return 'in progress'
            elif pd.notna(row['Start Date']) and row['Start Date'] > today:
                if current_status not in ['completed', 'overdue', 'in progress']:
                    return 'not started'
            return current_status if current_status != 'n/a' else 'not started'

        df['Status'] = df.apply(infer_status, axis=1)

        # --- Milestone Handling ---
        # For 'Milestone' types, if Start Date == End Date, make Start Date slightly before End Date
        # so Plotly can draw a very thin bar for it.
        milestone_mask = (df['Type'].fillna('').str.lower() == 'milestone') & (df['Start Date'] == df['End Date'])
        if not df[milestone_mask].empty:
            df.loc[milestone_mask, 'Start Date'] = df.loc[milestone_mask, 'End Date'] - timedelta(days=0.1)


        st.success("Excel file successfully uploaded and processed!")

        st.session_state.df = df

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.info("Please ensure it's a valid Excel file (.xlsx or .xls) and that required columns (ID, Name, Type, Start Date, End Date) are present and correctly formatted.")
        st.stop()

# --- Start of Main App Logic ---
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
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    selected_tab = st.sidebar.radio(
        "Go to:",
        ["Dashboard Overview", "Full Data Table"]
    )

    if selected_tab == "Dashboard Overview":
        # --- Dashboard Layout ---
        col1, col2 = st.columns([0.7, 0.3])

        with col1: # Left/Main Column for Timeline and Hierarchy
            st.header("Project Timeline (Gantt Chart)")
            st.markdown("Interact with the chart (zoom, pan) using the toolbar. Use sliders below to adjust date range.")

            # Timeline Granularity Selector
            time_scale_option = st.selectbox(
                "Select Timeline Scale:",
                ["Daily", "Weekly", "Monthly", "Quarterly", "Annually"],
                index=2 # Default to Monthly
            )

            # Date Range Slider for "Zoom"
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
                format="YYYY-MM-%d" # Using %d for daily precision as default for the slider
            )

            gantt_df = filtered_df[
                (filtered_df['Start Date'] <= pd.to_datetime(date_range_end)) &
                (filtered_df['End Date'] >= pd.to_datetime(date_range_start))
            ].copy() # Make a copy to avoid SettingWithCopyWarning


            if not gantt_df.empty:
                # --- Apply Project Sequence Sorting for Gantt Chart ---
                # Create a numerical rank for sorting based on Type
                type_rank_map = {
                    'program': 1,
                    'project': 1, # Both program and project at the same top level for sorting purposes
                    'sub-project': 2,
                    'task': 3,
                    'milestone': 4
                }
                gantt_df['Hierarchy_Rank'] = gantt_df['Type'].str.lower().map(type_rank_map).fillna(99) # Fillna for unknown types, giving them lowest priority

                # Sort the DataFrame before passing to px.timeline
                # Sort by Hierarchy_Rank, then Start Date, then Name
                gantt_df = gantt_df.sort_values(by=['Hierarchy_Rank', 'Start Date', 'Name'])

                gantt_df['Text'] = gantt_df.apply(
                    lambda row: f"{row['Name']} ({row['Type']})<br>Status: {row['Status'].capitalize()}<br>Progress: {row['Progress']:.0f}%", axis=1
                )

                # Define custom colors for types, including a distinct one for Milestones
                custom_colors = {
                    'Project': '#636EFA',
                    'Sub-Project': '#EF553B',
                    'Task': '#00CC96',
                    'Milestone': '#FFA15A', # A distinct color for milestones (orange-ish)
                    'Program': '#AB63FA'
                }
                # Ensure all types in gantt_df are in custom_colors, add a default if not already defined
                for item_type in gantt_df['Type'].unique():
                    if item_type not in custom_colors:
                        custom_colors[item_type] = '#19D3F3' # A default light blue

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
                        "Start Date": "|%Y-%m-%d", # Full date already here for hover
                        "End Date": "|%Y-%m-%d",   # Full date already here for hover
                        "Delivered Date": "|%Y-%m-%d",
                        "Progress": ":.0f%",
                        "Parent ID": True
                    },
                    color_discrete_map=custom_colors
                )
                fig.update_yaxes(autorange="reversed") # Keep reversed for Gantt standard (first item at top)
                fig.update_layout(height=600, showlegend=True)

                # Set X-axis tick format based on selected granularity
                if time_scale_option == "Daily":
                    fig.update_xaxes(tickformat='%Y-%m-%d')
                elif time_scale_option == "Weekly":
                    fig.update_xaxes(tickformat='%Y-W%W', dtick='W1') # dtick='W1' for weekly intervals
                elif time_scale_option == "Monthly":
                    fig.update_xaxes(tickformat='%Y-%m', dtick='M1') # dtick='M1' for monthly intervals
                elif time_scale_option == "Quarterly":
                    fig.update_xaxes(tickformat='%Y-Q%q', dtick='M3') # dtick='M3' for quarterly intervals (every 3 months)
                elif time_scale_option == "Annually":
                    fig.update_xaxes(tickformat='%Y', dtick='M12') # dtick='M12' for yearly intervals
                
                # --- Add Hovering Lines (Spikelines) ---
                fig.update_xaxes(
                    showspikes=True, # Corrected property name
                    spikemode="across", # Line across the plot area
                    spikesnap="cursor", # Snap spike to cursor's x-position
                    spikethickness=1,
                    spikecolor="rgba(0,0,0,0.5)" # Semi-transparent black
                )
                fig.update_yaxes(
                    showspikes=True, # Corrected property name
                    spikemode="across", # Line across the plot area
                    spikesnap="cursor", # Snap spike to cursor's y-position
                    spikethickness=1,
                    spikecolor="rgba(0,0,0,0.5)"
                )


                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No projects to display in the selected date range or with current filters.")

            st.header("📋 Hierarchical Project View")
            st.markdown("Expand projects to see their sub-projects, tasks, and milestones.")
            st.markdown("Items whose parent is filtered out or missing in the current view will appear at the top level with a ⚠️ note.")

            # Ensure ID and Parent ID are string and stripped (important for matching)
            # This was already done on df, but good practice to ensure on filtered_df as well
            filtered_df['ID'] = filtered_df['ID'].astype(str).str.strip()
            filtered_df['Parent ID'] = filtered_df['Parent ID'].astype(str).str.strip()

            # Re-create children_map based on the current filtered_df
            children_map = filtered_df.groupby('Parent ID')

            # Get all IDs present in the current filtered data
            existing_item_ids = set(filtered_df['ID'].unique())

            # Define top-level projects for display:
            # 1. Projects with no parent (Parent ID is empty string)
            # 2. Projects whose specified Parent ID does not exist within the current filtered_df's IDs (orphans in current view)
            top_level_projects_for_display = filtered_df[
                (filtered_df['Parent ID'] == '') |
                (~filtered_df['Parent ID'].isin(existing_item_ids))
            ].copy() # Make a copy

            # Recursive function to display children
            def display_children(parent_id, level=0):
                if parent_id in children_map.groups:
                    children = children_map.get_group(parent_id).copy() # Make a copy
                else:
                    children = pd.DataFrame()

                if children.empty:
                    return

                # Sort children by Hierarchy_Rank, then Start Date, then Name for consistent display order
                type_rank_map_children = { # Redefine for clarity, same as above
                    'program': 1,
                    'project': 1,
                    'sub-project': 2,
                    'task': 3,
                    'milestone': 4
                }
                children['Hierarchy_Rank'] = children['Type'].str.lower().map(type_rank_map_children).fillna(99)
                children = children.sort_values(by=['Hierarchy_Rank', 'Start Date', 'Name'])


                for index, row in children.iterrows():
                    prefix = "  " * level
                    progress_val = int(row['Progress'])
                    progress_bar_text = f" {progress_val}%"
                    
                    # Check if this item itself has children in the current filtered view
                    is_parent_node = row['ID'] in children_map.groups
                    
                    # Determine if it should be an expander
                    # An item should be an expander if it has children OR if its type is commonly a container (Project, Sub-Project, Program)
                    is_expandable_type = row['Type'].lower() in ['project', 'program', 'sub-project']
                    
                    if is_parent_node or is_expandable_type:
                        # Use a folder emoji for expandable nodes
                        expander_icon = "📁"
                        expander_label = f"{prefix}{expander_icon} **{row['Name']}** ({row['Type']}){progress_bar_text} - Status: {row.get('Status', 'N/A').title()}"
                        with st.expander(expander_label):
                            st.write(f"- **ID:** {row['ID']}")
                            st.write(f"- **Assigned To:** {row.get('Assigned To', 'N/A')}")
                            st.write(f"- **Start:**
