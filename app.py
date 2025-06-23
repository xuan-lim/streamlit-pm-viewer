import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic PM Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dynamic Project Management Viewer")

st.markdown("""
Upload your project management Excel file (.xlsx or .xls) to visualize and summarize your project data.
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
                st.error(f"Missing required column: `{col}`. Please check your Excel file.")
                st.stop()
            df[col] = df[col].astype(str).fillna('')

        st.success("Excel file successfully uploaded and processed!")

        # Store DataFrame in session state to avoid re-uploading
        st.session_state.df = df

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.info("Please ensure it's a valid Excel file (.xlsx or .xls) and that date columns are in a recognizable format.")
        st.stop()

# --- Main App Logic (only if DataFrame is loaded) ---
if 'df' in st.session_state:
    df = st.session_state.df

    st.sidebar.header("Navigation & Settings")
    selected_view = st.sidebar.selectbox(
        "Select View:",
        ["Project Hierarchy", "Alerts & Status", "Time-based Summaries"]
    )

    # --- Project Hierarchy View ---
    if selected_view == "Project Hierarchy":
        st.header("📋 Project & Milestone Hierarchy")
        st.markdown("Use the expanders below to view nested projects and milestones.")

        # Function to build and display hierarchy
        def display_children(parent_id, level=0):
            # Find direct children of the current parent_id
            children = df[df['Parent ID'] == parent_id]
            if children.empty:
                return

            for index, row in children.iterrows():
                prefix = "  " * level
                # Create a unique key for each expander to prevent issues
                expander_key = f"expander_{row['ID']}_{level}"

                # Decide if it's a project/sub-project that can have children or a leaf node
                if row['Type'] in ['Project', 'Sub-Project'] and row['ID'] in df['Parent ID'].values:
                    with st.expander(f"{prefix}📁 **{row['Name']}** ({row['Type']}) - Status: {row.get('Status', 'N/A')}", expanded=(level==0), key=expander_key):
                        st.write(f"- **ID:** {row['ID']}")
                        st.write(f"- **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                        st.write(f"- **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                        st.write(f"- **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                        st.write(f"- **Description:** (Add a 'Description' column to your Excel for more info)")
                        display_children(row['ID'], level + 1) # Recurse for children
                else: # It's a Task, Milestone, or a Project/Sub-Project without further children
                    st.markdown(f"{prefix}- **{row['Name']}** ({row['Type']})")
                    st.markdown(f"{prefix}  - **ID:** {row['ID']}")
                    st.markdown(f"{prefix}  - **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                    st.markdown(f"{prefix}  - **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                    st.markdown(f"{prefix}  - **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                    st.markdown(f"{prefix}  - **Status:** {row.get('Status', 'N/A')}")
                    st.markdown(f"{prefix}  - **Assigned To:** {row.get('Assigned To', 'N/A')}")
                    # You can still have a collapsed section for leaf nodes if they have more details
                    # with st.expander(f"{prefix}* {row['Name']} ({row['Type']}) - Status: {row.get('Status', 'N/A')}", expanded=False, key=expander_key):
                    #     st.write("More details here...")


        # Get top-level items (no parent)
        top_level_projects = df[df['Parent ID'].isna() | (df['Parent ID'] == '')]

        if not top_level_projects.empty:
            for index, row in top_level_projects.iterrows():
                # Ensure the top-level items are projects or similar entities
                if row['Type'] in ['Project', 'Program'] or row['ID'] in df['Parent ID'].values:
                    with st.expander(f"📌 **{row['Name']}** ({row['Type']}) - Status: {row.get('Status', 'N/A')}", expanded=True, key=f"top_expander_{row['ID']}"):
                        st.write(f"- **ID:** {row['ID']}")
                        st.write(f"- **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                        st.write(f"- **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                        st.write(f"- **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
                        display_children(row['ID'], 1)
                else:
                    st.markdown(f"**Top-level Item:** {row['Name']} ({row['Type']}) - Status: {row.get('Status', 'N/A')}")
                    st.markdown(f"  - **ID:** {row['ID']}")
                    st.markdown(f"  - **Start:** {row['Start Date'].strftime('%Y-%m-%d') if pd.notnull(row['Start Date']) else 'N/A'}")
                    st.markdown(f"  - **End:** {row['End Date'].strftime('%Y-%m-%d') if pd.notnull(row['End Date']) else 'N/A'}")
                    st.markdown(f"  - **Delivered:** {row['Delivered Date'].strftime('%Y-%m-%d') if pd.notnull(row['Delivered Date']) else 'N/A'}")
        else:
            st.info("No top-level projects found. Ensure 'Parent ID' is empty for your main projects.")

        st.markdown("---")
        st.subheader("Raw Data Table (for detailed inspection)")
        st.dataframe(df.style.highlight_max(axis=0, subset=['End Date']), use_container_width=True)


    # --- Alerts & Status View ---
    elif selected_view == "Alerts & Status":
        st.header("🔔 Alerts & Critical Status")
        today = pd.to_datetime(datetime.now().date()) # Only compare date part

        # Filter out completed items for alerts
        active_items = df[
            (df['Delivered Date'].isna()) | (df['Delivered Date'] > df['End Date'])
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if 'Status' in active_items.columns:
            active_items['Status'] = active_items['Status'].str.lower()


        st.subheader("❗ Overdue Items")
        overdue_items = active_items[
            (active_items['End Date'] < today) &
            (active_items['Status'].fillna('').isin(['not started', 'in progress', 'on hold', 'overdue', ''])) # Check relevant statuses
        ]
        if not overdue_items.empty:
            st.warning(f"You have {len(overdue_items)} overdue items!")
            st.dataframe(overdue_items[['Name', 'Type', 'End Date', 'Status', 'Assigned To', 'Parent ID']].sort_values(by='End Date'), use_container_width=True)
        else:
            st.info("🎉 No overdue items currently!")

        st.subheader("🔜 Items Due Soon (Next 7 Days)")
        due_soon_items = active_items[
            (active_items['End Date'] >= today) &
            (active_items['End Date'] <= today + timedelta(days=7)) &
            (active_items['Status'].fillna('').isin(['not started', 'in progress', 'on hold', ''])) # Check relevant statuses
        ]
        if not due_soon_items.empty:
            st.info(f"You have {len(due_soon_items)} items due in the next 7 days!")
            st.dataframe(due_soon_items[['Name', 'Type', 'End Date', 'Status', 'Assigned To', 'Parent ID']].sort_values(by='End Date'), use_container_width=True)
        else:
            st.info("Nothing due in the next 7 days.")

        st.subheader("✅ Completed Items (Last 30 Days)")
        # Filter for items delivered in the last 30 days or marked as completed
        completed_recently = df[
            (df['Delivered Date'].notna() & (df['Delivered Date'] >= today - timedelta(days=30))) |
            (df['Status'].fillna('') == 'completed' & (df['End Date'] >= today - timedelta(days=30)))
        ]
        if not completed_recently.empty:
            st.success(f"{len(completed_recently)} items recently completed or marked as complete!")
            st.dataframe(completed_recently[['Name', 'Type', 'Delivered Date', 'Status', 'Parent ID']].sort_values(by='Delivered Date', ascending=False), use_container_width=True)
        else:
            st.info("No items completed recently.")


    # --- Time-based Summaries View ---
    elif selected_view == "Time-based Summaries":
        st.header("📊 Time-based Summaries")

        summary_granularity = st.selectbox(
            "Select Summary Granularity:",
            ["Daily", "Weekly", "Monthly", "Quarterly", "Half-Yearly", "Yearly"]
        )

        st.markdown(f"Summarizing by **{summary_granularity}** based on **End Date**.")

        # Create a copy to avoid modifying original DataFrame
        summary_df = df.copy()
        # Drop rows where End Date is NaT as they can't be summarized by date
        summary_df = summary_df.dropna(subset=['End Date'])


        def get_time_period(date, granularity):
            if granularity == "Daily":
                return date.strftime('%Y-%m-%d')
            elif granularity == "Weekly":
                # Week starts on Monday (isocalendar week 1 starts with the first Monday of the year)
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

        summary_df['Time Period'] = summary_df['End Date'].apply(lambda x: get_time_period(x, summary_granularity))

        if not summary_df.empty:
            # Group by time period and count items
            grouped_summary = summary_df.groupby('Time Period').agg(
                Total_Items=('ID', 'count'),
                Completed_Items=('Delivered Date', lambda x: x.notna().sum()),
                Overdue_Items=('End Date', lambda x: ((x < datetime.now().date()) & (summary_df.loc[x.index, 'Delivered Date'].isna())).sum())
            ).reset_index()

            # Calculate "In Progress" as total minus completed and overdue (simplified, can be improved with 'Status' column)
            # This is a bit tricky without a 'Status' field for all entries, assuming 'not completed & not overdue' is 'in progress'
            # For better accuracy, use the 'Status' column in your Excel directly.
            grouped_summary['Remaining_Items'] = grouped_summary['Total_Items'] - grouped_summary['Completed_Items']

            # Sort by Time Period (this might need refinement for Quarter/Half-Year to sort correctly as strings)
            # For robust sorting, you might need to convert back to date objects or use a custom sort key.
            # For now, alphabetical will work for YYYY-MM, YYYY-MM-DD, YYYY-WXX
            grouped_summary = grouped_summary.sort_values(by='Time Period')

            st.dataframe(grouped_summary, use_container_width=True)

            # --- Visualization of Summary ---
            st.subheader("Summary Chart")
            # Using Melt for Stacked Bar Chart if needed, or simple bar chart for total items
            chart_data = grouped_summary[['Time Period', 'Total_Items', 'Completed_Items', 'Remaining_Items']]
            st.bar_chart(chart_data.set_index('Time Period'))

        else:
            st.info("No data available for the selected time summary after filtering for valid End Dates.")

    st.markdown("---")
    st.caption(f"App generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (New Taipei City time zone).")
