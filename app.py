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

        # Infer status for better alerts if status column is generic or missing/inconsistent
        today = pd.to_datetime(datetime.now().date())
        def infer_status(row):
            current_status = str(row.get('Status', 'n/a')).lower() # Use .get() and lower() for robustness

            if pd.notna(row['Delivered Date']) and pd.notna(row['End Date']) and row['Delivered Date'] <= row['End Date']:
                return 'completed'
            elif pd.notna(row['Delivered Date']) and pd.notna(row['End Date']) and row['Delivered Date'] > row['End Date']:
                 return 'completed (late)' # Mark as completed but late
            elif pd.notna(row['End Date']) and row['End Date'] < today:
                return 'overdue'
            elif pd.notna(row['Start Date']) and pd.notna(row['End Date']) and row['Start Date'] <= today and row['End Date'] >= today:
                if current_status not in ['completed', 'overdue']: # Don't overwrite explicit completed/overdue
                    return 'in progress'
            elif pd.notna(row['Start Date']) and row['Start Date'] > today:
                if current_status not in ['completed', 'overdue', 'in progress']: # Don't overwrite if already specified otherwise
                    return 'not started'
            # If no specific condition met, try to use existing status or default
            return current_status if current_status != 'n/a' else 'not started' # Default 'n/a' to 'not started'

        df['Status'] = df.apply(infer_status, axis=1)

        st.success("Excel file successfully uploaded and processed!")

        st.session_state.df = df

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.info("Please ensure it's a valid Excel file (.xlsx or .xls) and that required columns (ID, Name, Type, Start Date, End Date) are present and correctly formatted.")
        st.stop()

# --- Start of Main App Logic ---
# This 'if' statement MUST start at the very first character of the line (column 0).
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
            # Get min/max dates from filtered_df, convert to datetime.date
            # Provide sensible defaults if filtered_df is empty or dates are missing
            min_date_df = filtered_df['Start Date'].min()
            max_date_df = filtered_df['End Date'].max()

            # Ensure they are valid datetime objects before converting to date
            min_date = min_date_df.date() if pd.notna(min_date_df) else (datetime.now().date() - timedelta(days=90))
            max_date = max_date_df.date() if pd.notna(max_date_df) else (datetime.now().date() + timedelta(days=180))

            # Adjust min/max if the DataFrame is very short-term or future-only
            if min_date > max_date: # Handle cases where data might be single day or start > end for some reason
                min_date = max_date - timedelta(days=7) # Ensure a valid range

            # Ensure the initial value for the slider is also datetime.date objects
            # Use current date as a midpoint if the range is too wide or default
            today_date = datetime.now().date()
            default_slider_start = max(min_date, today_date - timedelta(days=30)) # Start 30 days before today, but not before min_date
            default_slider_end = min(max_date, today_date + timedelta(days=90))  # End 90 days after today, but not after max_date

            # Ensure default start is not after default end
            if default_slider_start > default_slider_end:
                 default_slider_start = default_slider_end - timedelta(days=7) # ensure at least 7 days range

            # THIS IS THE KEY FIX: Ensure all values are datetime.date objects for st.slider
            date_range_start, date_range_end = st.slider(
                "Select Date Range for Timeline:",
                min_value=min_date,
                max_value=max_date,
                value=(default_slider_start, default_slider_end),
                format="YYYY-MM-DD"
            )

            # Filter data for Gantt chart based on selected date range
            gantt_df = filtered_df[
                (filtered_df['Start Date'] <= pd.to_datetime(date_range_end)) & # Convert slider output back to Timestamp for filtering
                (filtered_df['End Date'] >= pd.to_datetime(date_range_start))
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

            # Create a dictionary for quick lookup of children
            # Ensure Parent ID is a string for consistent lookup
            filtered_df['Parent ID'] = filtered_df['Parent ID'].astype(str)
            
            # Group children by Parent ID
            children_map = filtered_df.groupby('Parent ID')

            def display_children(parent_id, level=0):
                # Check if this parent_id exists in the children_map
                if parent_id in children_map.groups:
                    children = children_map.get_group(parent_id)
                else:
                    children = pd.DataFrame() # No children found

                if children.empty:
                    return

                # Sort children for consistent display
                children = children.sort_values(by=['Start Date', 'Name'])

                for index, row in children.iterrows():
                    prefix = "  " * level # Indent based on level
                    
                    # Ensure Progress is an integer for display
                    progress_val = int(row['Progress'])
                    progress_bar_text = f" {progress_val}%"
                    
                    # Check if the current item itself is a parent (has children)
                    is_parent_node = row['ID'] in children_map.groups or row['Type'] in ['Project', 'Sub-Project']

                    if is_parent_node:
                        # Use expander for parent nodes (Projects, Sub-Projects, or items that have children)
                        with st.expander(f"{prefix}📁 **{row['Name']}** ({row['Type']}){progress_bar_text} - Status: {row.get('Status', 'N/A').title()}"):
                            st.write(f"- **ID:** {row['ID']}")
                            st.write(f"- **Assigned To:** {row.get('Assigned To', 'N/A')}")
                            st.write(f"- **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                            st.write(f"- **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                            st.write(f"- **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                            if progress_val > 0: # Only show progress bar if there's progress
                                st.progress(progress_val / 100) # Progress bar
                            display_children(row['ID'], level + 1) # Recurse for children
                    else:
                        # Display as plain markdown for leaf nodes (Tasks, Milestones without children)
                        status_emoji = "✅" if row['Status'] == 'completed' else "⏳" if row['Status'] == 'in progress' else "🔴" if row['Status'] == 'overdue' else "⚪"
                        st.markdown(f"{prefix}{status_emoji} **{row['Name']}** ({row['Type']}){progress_bar_text}")
                        st.markdown(f"{prefix}  - **ID:** {row['ID']}")
                        st.markdown(f"{prefix}  - **Assigned To:** {row.get('Assigned To', 'N/A')}")
                        st.markdown(f"{prefix}  - **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                        st.markdown(f"{prefix}  - **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                        st.markdown(f"{prefix}  - **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                        st.markdown(f"{prefix}  - **Status:** {row.get('Status', 'N/A').title()}")
                        if progress_val > 0: # Only show progress bar if there's progress
                            st.progress(progress_val / 100) # Progress bar

            # Get top-level items (no parent_id, and must be in filtered_df)
            # Filter out items that are themselves 'Parent ID's to avoid double counting or starting nested trees incorrectly.
            # Only display top-level items if their 'ID' is NOT in the 'Parent ID' column (i.e., they are not children of anything).
            all_parent_ids_in_data = set(filtered_df['Parent ID'].unique())
            top_level_projects = filtered_df[
                (filtered_df['Parent ID'] == '') | (filtered_df['Parent ID'].isna())
            ].sort_values(by=['Start Date', 'Name'])

            # If there are items whose Parent ID is not in the 'ID' column, they are also effectively top-level
            # This handles cases where a sub-project might be loaded without its main parent.
            potential_orphan_parents = filtered_df[
                (~filtered_df['Parent ID'].isin(filtered_df['ID'])) & # Parent ID is not a valid ID
                (filtered_df['Parent ID'] != '') & (filtered_df['Parent ID'].notna()) # And Parent ID is not empty/null
            ]
            # Add these to top_level_projects if they are not already there
            top_level_projects = pd.concat([top_level_projects, potential_orphan_parents]).drop_duplicates(subset=['ID']).sort_values(by=['Start Date', 'Name'])


            if not top_level_projects.empty:
                for index, row in top_level_projects.iterrows():
                    progress_val = int(row['Progress'])
                    progress_bar_text = f" {progress_val}%"
                    
                    is_parent_node = row['ID'] in children_map.groups or row['Type'] in ['Project', 'Program']

                    if is_parent_node:
                        with st.expander(f"📌 **{row['Name']}** ({row['Type']}){progress_bar_text} - Status: {row.get('Status', 'N/A').title()}", expanded=True):
                            st.write(f"- **ID:** {row['ID']}")
                            st.write(f"- **Assigned To:** {row.get('Assigned To', 'N/A')}")
                            st.write(f"- **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                            st.write(f"- **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                            st.write(f"- **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                            if progress_val > 0:
                                st.progress(progress_val / 100)
                            display_children(row['ID'], 1)
                    else: # A top-level item that doesn't act as a parent (e.g., a standalone task, or a type not configured as parent)
                        status_emoji = "✅" if row['Status'] == 'completed' else "⏳" if row['Status'] == 'in progress' else "🔴" if row['Status'] == 'overdue' else "⚪"
                        st.markdown(f"**{status_emoji} {row['Name']}** ({row['Type']}){progress_bar_text} - Status: {row.get('Status', 'N/A').title()}")
                        st.markdown(f"  - **ID:** {row['ID']}")
                        st.markdown(f"  - **Assigned To:** {row.get('Assigned To', 'N/A')}")
                        st.markdown(f"  - **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                        st.markdown(f"  - **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                        st.markdown(f"  - **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                        if progress_val > 0:
                            st.progress(progress_val / 100)

            else:
                st.info("No top-level projects found or all filtered out. Ensure 'Parent ID' is empty for your main projects.")


        with col2: # Right Column for Alerts and Summaries
            st.header("🔔 Alerts")
            today_date_only = datetime.now().date() # Use native date for comparisons

            # Filter active items for alerts (not completed or overdue due to delayed delivery)
            # Ensure 'End Date' is a datetime.date object for comparison, if it came from pandas.Timestamp
            alert_items = filtered_df[
                ((filtered_df['Status'].isin(['in progress', 'not started', 'on hold', 'overdue'])) |
                 (filtered_df['Delivered Date'].notna() & (filtered_df['Delivered Date'].dt.date > filtered_df['End Date'].dt.date))) &
                (filtered_df['End Date'].notna()) # Ensure End Date exists for comparison
            ].copy()

            st.subheader("❗ Overdue Items")
            # Sort by End Date ascending so earliest overdue appear first
            overdue_items = alert_items[alert_items['End Date'].dt.date < today_date_only].sort_values(by='End Date')
            if not overdue_items.empty:
                for idx, item in overdue_items.head(5).iterrows(): # Show top 5 overdue
                    st.error(f"**{item['Name']}** ({item['Type']}) is overdue! (Due: {item['End Date'].strftime('%Y-%m-%d')})")
            else:
                st.info("🎉 No overdue items currently!")

            st.subheader("🔜 Due Soon (Next 7 Days)")
            # Sort by End Date ascending so earliest due appear first
            due_soon_items = alert_items[
                (alert_items['End Date'].dt.date >= today_date_only) &
                (alert_items['End Date'].dt.date <= today_date_only + timedelta(days=7))
            ].sort_values(by='End Date')

            if not due_soon_items.empty:
                for idx, item in due_soon_items.head(5).iterrows(): # Show top 5 due soon
                    st.warning(f"**{item['Name']}** ({item['Type']}) due in {(item['End Date'].dt.date - today_date_only).days} days! (End: {item['End Date'].strftime('%Y-%m-%d')})")
            else:
                st.info("Nothing due in the next 7 days.")

            st.subheader("✅ Recently Completed (Last 30 Days)")
            # Sort by Delivered Date descending for most recent first
            completed_recently = filtered_df[
                (filtered_df['Delivered Date'].notna()) &
                (filtered_df['Delivered Date'].dt.date >= today_date_only - timedelta(days=30))
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
                (filtered_df['Status'].isin(['completed', 'completed (late)']))
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


            def get_time_period(date_ts, granularity): # Accepts Timestamp
                # Convert to Python date object for consistent time period calculation
                date = date_ts.date()
                if granularity == "Daily":
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

                # Group by Time Period and get counts for each status
                status_counts_by_time = summary_df_filtered.groupby(['Time Period', 'Status'])['ID'].count().unstack(fill_value=0)

                # Rename columns for clarity in the summary table
                status_counts_by_time.columns = [col.replace(" ", "_").title() + "_Items" for col in status_counts_by_time.columns]

                # Ensure 'Total_Items' is also included by summing up across rows
                status_counts_by_time['Total_Items'] = status_counts_by_time.sum(axis=1)

                grouped_summary = status_counts_by_time.reset_index()

                # Reorder columns to have Total_Items first, then others alphabetically for consistency
                cols_order = ['Time Period', 'Total_Items'] + sorted([col for col in grouped_summary.columns if col not in ['Time Period', 'Total_Items']])
                grouped_summary = grouped_summary[cols_order]

                grouped_summary = grouped_summary.sort_values(by='Time Period')

                st.dataframe(grouped_summary, use_container_width=True)

                st.subheader("Items by Status over Time")
                # Prepare data for stacked bar chart
                chart_data_melted = grouped_summary.melt(
                    id_vars='Time Period',
                    value_vars=[col for col in grouped_summary.columns if 'Items' in col and col != 'Total_Items'],
                    var_name='Status Type',
                    value_name='Count'
                )
                fig_summary = px.bar(
                    chart_data_melted,
                    x='Time Period',
                    y='Count',
                    color='Status Type',
                    title=f'Item Status by {summary_granularity}',
                    barmode='stack',
                    color_discrete_map={
                        'Completed_Items': 'green',
                        'Completed (late)_Items': 'darkgreen', # Distinguish late completions
                        'In_Progress_Items': 'blue',
                        'Not_Started_Items': 'grey',
                        'Overdue_Items': 'red'
                    }
                )
                st.plotly_chart(fig_summary, use_container_width=True)

            else:
                st.info("No data available for the selected time summary.")

    # This 'elif' statement MUST be aligned with the 'if' statement for 'Dashboard Overview'
    elif selected_tab == "Full Data Table":
        st.header("Raw Project Data")
        st.markdown("This table shows all data from your uploaded Excel file after initial processing.")
        st.dataframe(filtered_df, use_container_width=True)


    st.markdown("---")
    st.caption(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Current time zone).")
