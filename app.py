import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

# --- Page Configuration ---
st.set_page_config(
    page_title="PM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        padding: 1rem 0;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .alert-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .alert-danger {
        background-color: #f8d7da;
        border-color: #f1aeb5;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .project-hierarchy {
        font-family: 'Courier New', monospace;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
    /* Enhance Streamlit buttons */
    div.stButton > button:first-child {
        background-color: #667eea;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
class ProjectSequencer:
    """Handles project sequencing and hierarchy management"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.hierarchy_map = self._build_hierarchy_map()

    def _build_hierarchy_map(self) -> Dict[str, List[str]]:
        """Build a map of parent-child relationships"""
        hierarchy = {}
        for _, row in self.df.iterrows():
            parent_id = str(row.get('Parent ID', '')).strip()
            child_id = str(row['ID']).strip()

            if parent_id and parent_id != 'nan':
                if parent_id not in hierarchy:
                    hierarchy[parent_id] = []
                hierarchy[parent_id].append(child_id)
        return hierarchy

    def get_project_depth(self, project_id: str) -> int:
        """Calculate the hierarchical depth of a project"""
        project_id = str(project_id).strip()
        project_row = self.df[self.df['ID'] == project_id]

        if project_row.empty:
            return 0

        parent_id = str(project_row.iloc[0].get('Parent ID', '')).strip()
        if not parent_id or parent_id == 'nan':
            return 0

        # Recursively get depth of parent
        return 1 + self.get_project_depth(parent_id)

    def get_sequence_order(self) -> pd.DataFrame:
        """Return DataFrame with sequence ordering for hierarchy display"""
        df_with_sequence = self.df.copy()

        # Add depth information
        df_with_sequence['Hierarchy_Depth'] = df_with_sequence['ID'].apply(self.get_project_depth)

        # Create type priority mapping to order different item types
        type_priority = {
            'program': 1,
            'project': 2,
            'sub-project': 3,
            'task': 4,
            'milestone': 5
        }

        # Apply priority, default to a high number for unknown types
        df_with_sequence['Type_Priority'] = df_with_sequence['Type'].str.lower().map(type_priority).fillna(99)

        # Sort by multiple criteria for proper sequencing in Gantt and Hierarchy views
        df_with_sequence = df_with_sequence.sort_values([
            'Hierarchy_Depth',
            'Type_Priority',
            'Start Date',
            'Name'
        ], na_position='last') # Ensure NaNs are at the end

        return df_with_sequence

class DataProcessor:
    """Handles data cleaning and preprocessing"""

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the uploaded DataFrame.
        Ensures correct data types and handles missing values.
        """
        df_clean = df.copy()

        # Clean date columns, coercing errors will turn invalid dates into NaT
        date_cols = ['Start Date', 'End Date', 'Delivered Date']
        for col in date_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            else:
                df_clean[col] = pd.NaT # Add missing date columns as NaT

        # Clean ID columns, convert to string and strip whitespace
        df_clean['ID'] = df_clean['ID'].astype(str).str.strip()
        if 'Parent ID' in df_clean.columns:
            df_clean['Parent ID'] = df_clean['Parent ID'].astype(str).str.strip().fillna('')
        else:
            df_clean['Parent ID'] = '' # Add missing Parent ID column as empty string

        # Ensure required text columns exist and are cleaned
        for col in ['Name', 'Type']:
            if col not in df_clean.columns:
                # If a core column is missing, raise a specific error
                raise ValueError(f"Missing required column: '{col}'. Please ensure your Excel file has 'ID', 'Name', 'Type', 'Start Date', 'End Date'.")
            df_clean[col] = df_clean[col].astype(str).fillna('')

        # Handle Progress column: if missing, infer from 'Delivered Date'
        if 'Progress' not in df_clean.columns:
            df_clean['Progress'] = df_clean['Delivered Date'].apply(
                lambda x: 100 if pd.notna(x) else 0
            )
        else:
            # Convert to numeric, handle errors by filling with 0, and clip to 0-100 range
            df_clean['Progress'] = pd.to_numeric(df_clean['Progress'], errors='coerce').fillna(0)
            df_clean['Progress'] = df_clean['Progress'].clip(0, 100)

        # Handle Status column: if missing, set to 'N/A'
        if 'Status' not in df_clean.columns:
            df_clean['Status'] = 'N/A'
        else:
            df_clean['Status'] = df_clean['Status'].astype(str).str.lower().fillna('n/a')

        return df_clean

    @staticmethod
    def infer_status(df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer project status based on dates and current time.
        Prioritizes 'completed' states, then 'overdue', 'in progress', 'not started'.
        """
        df_status = df.copy()
        today = datetime.now().date()

        def calculate_status(row):
            current_status = str(row.get('Status', 'n/a')).lower() # Preserve existing status if it makes sense

            # Convert Series elements to date objects for consistent comparison
            start_date_dt = row['Start Date'].date() if pd.notna(row['Start Date']) else None
            end_date_dt = row['End Date'].date() if pd.notna(row['End Date']) else None
            delivered_date_dt = row['Delivered Date'].date() if pd.notna(row['Delivered Date']) else None

            # 1. Completed cases (highest priority)
            if delivered_date_dt is not None:
                if end_date_dt is not None and delivered_date_dt > end_date_dt:
                    return 'completed (late)'
                else:
                    return 'completed'

            # 2. Overdue case
            if end_date_dt is not None and end_date_dt < today:
                return 'overdue'

            # 3. In progress case
            if start_date_dt is not None and end_date_dt is not None and \
               start_date_dt <= today <= end_date_dt:
                return 'in progress'

            # 4. Not started case (future start date)
            if start_date_dt is not None and start_date_dt > today:
                return 'not started'
            
            # If nothing else matches and a status was provided, use it. Otherwise, default to 'not started'.
            return current_status if current_status not in ['n/a', ''] else 'not started'


        df_status['Status'] = df_status.apply(calculate_status, axis=1)
        return df_status

class DashboardVisualizer:
    """Handles dashboard visualizations"""

    def create_enhanced_gantt(self, df: pd.DataFrame, date_range: Tuple[datetime, datetime]) -> go.Figure:
        """
        Create an enhanced Gantt chart with a scalable timeline, range slider, grid, and today line.
        """
        if df.empty:
            return go.Figure().add_annotation(text="No data to display in Gantt chart.", xref="paper", yref="paper", showarrow=False)

        # Filter DataFrame based on the selected master date range
        gantt_df = df[
            (df['Start Date'] <= pd.to_datetime(date_range[1])) &
            (df['End Date'] >= pd.to_datetime(date_range[0]))
        ].copy()

        if gantt_df.empty:
            # Return an empty figure with a message if no items are in the range
            st.info("No project items fall within the selected date range.")
            return go.Figure().add_annotation(text="No items in selected date range.", xref="paper", yref="paper", showarrow=False)

        # For milestones, make them appear as points or very short bars
        milestone_mask = (gantt_df['Type'].str.lower() == 'milestone')
        # Ensure milestones have a small duration for visibility, e.g., 1 day
        gantt_df.loc[milestone_mask, 'End Date'] = gantt_df.loc[milestone_mask, 'Start Date'] + timedelta(days=1)
        # Handle cases where start date is NaT for milestones.
        gantt_df = gantt_df[gantt_df['Start Date'].notna()]

        # Apply sequencing for better visual hierarchy
        sequencer = ProjectSequencer(gantt_df)
        gantt_df = sequencer.get_sequence_order()

        fig = go.Figure()

        # Define consistent colors for different item types
        color_map = {
            'Program': '#1f77b4', 'Project': '#ff7f0e', 'Sub-Project': '#2ca02c',
            'Task': '#d62728', 'Milestone': '#9467bd', 'On Hold': '#ffc107', 'N/A': '#999999'
        }

        # Add a trace for each item
        for index, row in gantt_df.iterrows():
            # Calculate the duration in milliseconds for the x-axis
            duration_ms = (row['End Date'] - row['Start Date']).total_seconds() * 1000
            
            # Ensure duration is non-negative
            if duration_ms < 0:
                duration_ms = 0
            
            fig.add_trace(go.Bar(
                # Use item Name on Y-axis
                y=[row['Name']],
                # X-axis represents the duration starting from 'Start Date'
                x=[duration_ms],
                base=[row['Start Date']],
                orientation='h', # Horizontal bars
                marker_color=color_map.get(row['Type'], '#17becf'), # Get color from map, default if not found
                text=f"{row['Progress']:.0f}%", # Display progress as text on bar
                textposition='inside', # Position text inside bars
                # Custom hover template for rich information on hover
                hovertemplate=(
                    f"<b>{row['Name']}</b><br>"
                    f"Type: {row['Type']}<br>"
                    "Start: %{base|%Y-%m-%d}<br>" # Format start date
                    "End: %{x|%Y-%m-%d}<br>"     # Format end date (from base + duration)
                    f"Progress: {row['Progress']:.0f}%<br>"
                    f"Status: {row['Status'].title()}<extra></extra>" # Remove extra trace info
                ),
                legendgroup=row['Type'], # Group legend items by Type
                showlegend=(row['Type'] not in [t.name for t in fig.data]) # Show legend only once per type
            ))
            
        # Add a vertical line for today's date
        fig.add_vline(
            x=datetime.now(), # Today's date and time
            line_width=2,
            line_dash="solid",
            line_color="red",
            annotation_text="Today", # Add a subtle annotation
            annotation_position="top right"
        )

        fig.update_layout(
            title_text='Enhanced Project Timeline',
            height=max(700, len(gantt_df) * 35), # Adjust height dynamically based on number of items
            barmode='overlay', # Bars overlap, useful for progress if implemented differently
            legend=dict(title="Item Type", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='closest', # Show hover info for the closest data point
            xaxis=dict(
                title='Timeline', type='date', # X-axis is a date type
                rangeselector=dict( # Add range selector buttons for quick date filtering
                    buttons=list([
                        dict(count=7, label="Week", step="day", stepmode="backward"),
                        dict(count=1, label="Month", step="month", stepmode="backward"),
                        dict(count=3, label="Quarter", step="month", stepmode="backward"),
                        dict(count=6, label="Half-Year", step="month", stepmode="backward"),
                        dict(count=1, label="Year", step="year", stepmode="todate"),
                        dict(step="all", label="All")
                    ])
                ),
                rangeslider=dict(visible=True), # Show a range slider at the bottom
                showspikes=True, spikemode='across', spikesnap='cursor', # Spikes for better interaction
                spikethickness=1, spikedash='dot',
                showgrid=True, gridcolor='LightGrey', gridwidth=1 # Grid for readability
            ),
            yaxis=dict(
                title='Projects',
                autorange="reversed", # Ensure top items are higher on the chart
                showgrid=True, gridcolor='LightGrey', gridwidth=1
            ),
            margin=dict(l=150, r=50, t=80, b=80) # Adjust margins for better layout
        )
        return fig

    def create_status_summary_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a pie chart showing the distribution of project statuses."""
        if df.empty:
            return go.Figure().add_annotation(text="No data for status chart.", xref="paper", yref="paper", showarrow=False)

        status_counts = df['Status'].value_counts()
        colors = {
            'completed': '#28a745',          # Green
            'completed (late)': '#20c997',   # Teal-green
            'in progress': '#007bff',        # Blue
            'not started': '#6c757d',        # Grey
            'overdue': '#dc3545',            # Red
            'on hold': '#ffc107',            # Yellow
            'n/a': '#17a2b8'                 # Info blue
        }
        fig = go.Figure(data=[
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                marker_colors=[colors.get(status.lower(), '#17a2b8') for status in status_counts.index],
                textinfo='label+percent', # Show label and percentage
                textposition='auto',      # Automatically position text
                hole=0.3,                 # Create a donut chart
                hoverinfo='label+value+percent' # Show detailed info on hover
            )
        ])
        fig.update_layout(
            title_text='Project Status Distribution',
            height=400,
            showlegend=True,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        return fig

# --- Main Application ---
def main():
    st.markdown('<div class="main-header"><h1>📊 Project Management Dashboard</h1></div>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Settings")

    st.markdown("### 📁 Upload Project Data")
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx or .xls)", type=["xlsx", "xls"],
        help="Upload your project management Excel file with columns: ID, Name, Type, Start Date, End Date, Parent ID (optional), Progress (optional), Delivered Date (optional), Status (optional)"
    )

    # Initialize df in session_state if not already present
    if 'df' not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if uploaded_file is not None:
        try:
            with st.spinner("Processing your file..."):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                df_clean = DataProcessor.clean_dataframe(df)
                df_processed = DataProcessor.infer_status(df_clean)
            st.success("✅ File successfully processed!")
            st.session_state.df = df_processed
        except ValueError as ve:
            st.error(f"❌ Data Error: {str(ve)}")
            st.info("Please ensure your Excel file contains the required columns: 'ID', 'Name', 'Type', 'Start Date', 'End Date'.")
            st.session_state.df = pd.DataFrame() # Clear potentially problematic data
            return
        except Exception as e:
            st.error(f"❌ An unexpected error occurred during file processing: {str(e)}")
            st.info("Please check the file format and column names.")
            st.session_state.df = pd.DataFrame() # Clear potentially problematic data
            return

    if not st.session_state.df.empty:
        df = st.session_state.df

        st.sidebar.header("🔍 Filters")
        
        # Determine top-level projects, handling cases where Parent ID might not exist in ID column
        top_level_projects = df[
            (df['Parent ID'].isin(['', 'nan'])) | (~df['Parent ID'].isin(df['ID']))
        ]['Name'].unique()
        
        selected_projects = st.sidebar.multiselect("Filter by Top-Level Project/Program:", top_level_projects, help="Select specific top-level projects or programs to focus on. This will include their children.")
        
        # If specific projects are selected, filter the DataFrame to include them and their descendants
        filtered_df = df.copy()
        if selected_projects:
            # Get IDs of selected top-level projects
            selected_ids = df[df['Name'].isin(selected_projects)]['ID'].tolist()
            
            all_included_ids = set(selected_ids)
            current_level_ids = set(selected_ids)

            # Iteratively find all descendants
            while True:
                # Find children whose parent IDs are in the current level's IDs
                children = filtered_df[filtered_df['Parent ID'].isin(current_level_ids)]['ID'].tolist()
                new_children = set(children) - all_included_ids # Find only new children
                
                if not new_children:
                    break # No new children found, stop
                
                all_included_ids.update(new_children) # Add new children to the overall set
                current_level_ids = new_children # Set current level to the newly found children for next iteration

            # Filter the DataFrame to include only the selected projects and their descendants
            filtered_df = filtered_df[filtered_df['ID'].isin(all_included_ids)]

        # Further filter by Type and Status
        all_types = sorted(df['Type'].unique())
        selected_types = st.sidebar.multiselect("Filter by Type:", all_types, default=all_types)
        
        all_statuses = sorted(df['Status'].unique())
        selected_statuses = st.sidebar.multiselect("Filter by Status:", all_statuses, default=all_statuses)

        filtered_df = filtered_df[filtered_df['Type'].isin(selected_types) & filtered_df['Status'].isin(selected_statuses)]
        
        st.sidebar.header("📋 Navigation")
        tab = st.sidebar.radio("Select View:", ["📊 Dashboard", "📈 Analytics", "📋 Data Table", "🏗️ Project Hierarchy"])

        # Display content based on selected tab
        if tab == "📊 Dashboard":
            show_dashboard(filtered_df)
        elif tab == "📈 Analytics":
            show_analytics(filtered_df)
        elif tab == "📋 Data Table":
            show_data_table(filtered_df)
        elif tab == "🏗️ Project Hierarchy":
            show_project_hierarchy(filtered_df)
    else:
        st.info("Please upload an Excel file to get started with the dashboard.")

def show_dashboard(df: pd.DataFrame):
    """Main dashboard view with a full-width layout for the Gantt chart."""
    st.header("📅 Project Timeline")

    if not df.empty and 'Start Date' in df.columns and 'End Date' in df.columns:
        # Calculate min and max dates from available data for default range
        min_date_available = df['Start Date'].min()
        max_date_available = df['End Date'].max()

        # Handle NaT values if min/max dates are not available
        min_date_val = min_date_available.date() if pd.notna(min_date_available) else (datetime.now() - timedelta(days=365)).date()
        max_date_val = max_date_available.date() if pd.notna(max_date_available) else (datetime.now() + timedelta(days=365)).date()

        date_range_selection = st.date_input(
            "Select a master date range to filter items for the Gantt chart:",
            value=[min_date_val, max_date_val],
            min_value=min_date_val,
            max_value=max_date_val,
            help="This filters the project items displayed in the Gantt chart. Use the chart's internal controls (rangeselector, rangeslider) to zoom in on specific periods."
        )

        # Ensure two dates are selected from the date_input widget
        if len(date_range_selection) == 2:
            # Convert selected dates to datetime objects for comparison with DataFrame columns
            gantt_date_start = datetime.combine(date_range_selection[0], datetime.min.time())
            gantt_date_end = datetime.combine(date_range_selection[1], datetime.max.time())
            
            with st.spinner("Generating Gantt chart..."):
                visualizer = DashboardVisualizer()
                gantt_fig = visualizer.create_enhanced_gantt(df, (gantt_date_start, gantt_date_end))
                st.plotly_chart(gantt_fig, use_container_width=True)
        else:
            st.info("Please select both a start and end date for the master filter to display the Gantt chart.")
    else:
        st.info("Upload data with 'Start Date' and 'End Date' columns to see the timeline. No data available based on current filters or missing date columns.")
            
    st.markdown("---")
    
    # Create two columns for quick stats and alerts
    col1, col2 = st.columns(2)
    with col1:
        st.header("📊 Quick Stats")
        total_items = len(df)
        completed_items = len(df[df['Status'].str.contains('completed', case=False, na=False)])
        overdue_items = len(df[df['Status'] == 'overdue'])
        in_progress_items = len(df[df['Status'] == 'in progress'])

        # Display metrics in a row of four columns
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Items", total_items)
        c2.metric("Completed", completed_items, delta=f"{(completed_items/total_items*100):.1f}%" if total_items > 0 else "0%")
        c3.metric("In Progress", in_progress_items)
        c4.metric("Overdue", overdue_items, delta_color="inverse", delta=f"{overdue_items}")
        
        # Display status distribution pie chart
        if not df.empty:
            visualizer = DashboardVisualizer()
            status_fig = visualizer.create_status_summary_chart(df)
            st.plotly_chart(status_fig, use_container_width=True)
        else:
            st.info("No data available to show status distribution.")
            
    with col2:
        show_alerts(df) # Display alerts in the second column

def show_analytics(df: pd.DataFrame):
    """Displays various analytical charts about the project data."""
    if df.empty:
        st.info("No data available for analytics based on current filters.")
        return
    st.header("📈 Project Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        if 'Progress' in df.columns and not df['Progress'].empty:
            fig_progress = px.histogram(df, x='Progress', nbins=20, title='Progress Distribution', labels={'Progress': 'Progress (%)', 'count': 'Number of Items'},
                                        color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig_progress, use_container_width=True)
        else:
            st.info("No 'Progress' data available for this chart.")
            
    with col2:
        if 'Type' in df.columns and not df['Type'].empty:
            type_counts = df['Type'].value_counts()
            fig_types = px.bar(x=type_counts.index, y=type_counts.values, title='Items by Type', labels={'x': 'Type', 'y': 'Count'},
                               color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_types, use_container_width=True)
        else:
            st.info("No 'Type' data available for this chart.")
            
    # Project Duration vs. Start Date
    if 'Start Date' in df.columns and 'End Date' in df.columns:
        # Filter out rows with NaT for start or end dates
        df_timeline = df.dropna(subset=['Start Date', 'End Date']).copy()
        if not df_timeline.empty:
            df_timeline['Duration'] = (df_timeline['End Date'] - df_timeline['Start Date']).dt.days
            # Filter out negative durations (invalid data)
            df_timeline = df_timeline[df_timeline['Duration'] >= 0] 
            
            if not df_timeline.empty:
                fig_duration = px.scatter(df_timeline, x='Start Date', y='Duration', color='Type', size='Progress', 
                                          hover_data={'Name': True, 'Status': True, 'Progress': ':.0f%', 'Duration': ':.0f days'}, 
                                          title='Project Duration vs. Start Date',
                                          color_discrete_sequence=px.colors.qualitative.Dark24)
                st.plotly_chart(fig_duration, use_container_width=True)
            else:
                st.info("No valid duration data to display for 'Project Duration vs. Start Date' chart.")
        else:
            st.info("No valid 'Start Date' and 'End Date' data to display 'Project Duration vs. Start Date' chart.")

def show_data_table(df: pd.DataFrame):
    """Displays the raw project data in a table format with download option."""
    st.header("📋 Project Data Table")
    if df.empty:
        st.info("No data to display based on current filters.")
        return
    
    # Download button for CSV export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download Data as CSV", data=csv, file_name=f"project_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    
    # Display DataFrame with enhanced column configurations
    st.dataframe(df, use_container_width=True, column_config={
        "Progress": st.column_config.ProgressColumn("Progress", help="Project completion percentage", format="%.0f%%", min_value=0, max_value=100),
        "Start Date": st.column_config.DateColumn("Start Date", format="YYYY-MM-DD"),
        "End Date": st.column_config.DateColumn("End Date", format="YYYY-MM-DD"),
        "Delivered Date": st.column_config.DateColumn("Delivered Date", format="YYYY-MM-DD"),
        "Name": st.column_config.TextColumn("Project/Task Name", help="Name of the project or task"),
        "Type": st.column_config.TextColumn("Type", help="Type of item (e.g., Project, Task, Milestone)"),
        "Status": st.column_config.TextColumn("Status", help="Current status of the item"),
        "ID": st.column_config.TextColumn("ID", help="Unique identifier for the item"),
        "Parent ID": st.column_config.TextColumn("Parent ID", help="ID of the parent item in the hierarchy")
    })

def show_project_hierarchy(df: pd.DataFrame):
    """Displays the project hierarchy in a tree-like structure."""
    st.header("🏗️ Project Hierarchy")
    if df.empty:
        st.info("No data to display based on current filters.")
        return
    
    sequencer = ProjectSequencer(df)
    # Get the sorted DataFrame to ensure consistent ordering
    full_sorted_df = sequencer.get_sequence_order()
    
    # Filter the sorted DataFrame to only include items relevant to the current filters
    df_to_display = full_sorted_df[full_sorted_df['ID'].isin(df['ID'])].copy()

    # Identify top-level items that either have no parent or their parent is not in the filtered dataset
    top_level = df_to_display[
        (df_to_display['Parent ID'].isin(['', 'nan'])) | (~df_to_display['Parent ID'].isin(df_to_display['ID']))
    ].sort_values(['Name']).copy() # Sort top-level items by name for consistency

    def display_project_tree(parent_df: pd.DataFrame, level: int = 0):
        """
        Recursively displays the project hierarchy.
        `parent_df`: DataFrame subset containing items at the current level to display.
        `level`: Current indentation level for visualization.
        """
        for _, row in parent_df.iterrows():
            indent = "&nbsp;&nbsp;&nbsp;&nbsp;" * level # HTML non-breaking spaces for indentation
            
            # Emoji for status visualization
            status_emoji = {
                'completed': '✅', 'completed (late)': '✅', 'in progress': '🔄',
                'overdue': '🔴', 'not started': '⏳', 'on hold': '⏸️'
            }.get(row['Status'], '⚪')
            
            # Simple text-based progress bar
            progress_bar = "█" * int(row['Progress'] // 10) + "░" * (10 - int(row['Progress'] // 10))
            
            # Format dates for display
            start_date_str = row['Start Date'].strftime('%Y-%m-%d') if pd.notna(row['Start Date']) else 'N/A'
            end_date_str = row['End Date'].strftime('%Y-%m-%d') if pd.notna(row['End Date']) else 'N/A'

            # Display using markdown with custom CSS class for styling
            st.markdown(f"""
            <div class="project-hierarchy">
                <pre style="margin: 0; padding: 0; white-space: pre-wrap;">{indent}{status_emoji} <strong>{row['Name']}</strong> ({row['Type']})</pre>
                <pre style="margin: 0; padding: 0; white-space: pre-wrap;">{indent}&nbsp; Progress: {progress_bar} {row['Progress']:.0f}%</pre>
                <pre style="margin: 0; padding: 0; white-space: pre-wrap;">{indent}&nbsp; Status: {row['Status'].title()}</pre>
                <pre style="margin: 0; padding: 0; white-space: pre-wrap;">{indent}&nbsp; Duration: {start_date_str} &rarr; {end_date_str}</pre>
            </div>""", unsafe_allow_html=True)
            
            # Find children of the current item within the *filtered* data to display
            children = df_to_display[df_to_display['Parent ID'] == row['ID']].sort_values(['Name'])
            if not children.empty:
                display_project_tree(children, level + 1) # Recurse for children
                
    display_project_tree(top_level) # Start the recursion from top-level items

def show_alerts(df: pd.DataFrame):
    """Displays various alerts based on project status and dates."""
    st.header("🚨 Alerts")
    today = datetime.now().date()

    # Overdue Items
    overdue = df[df['Status'] == 'overdue']
    if not overdue.empty:
        st.markdown(f'<div class="alert-danger"><strong>⚠️ {len(overdue)} Overdue Items</strong></div>', unsafe_allow_html=True)
        with st.expander("View Overdue Items"):
            for _, item in overdue.iterrows():
                due_date = item['End Date'].strftime('%Y-%m-%d') if pd.notna(item['End Date']) else 'N/A'
                st.markdown(f"• **{item['Name']}** (ID: {item['ID']}) - Due: {due_date}")

    # Items Due This Week
    week_ahead = today + timedelta(days=7)
    # Corrected comparison: compare Timestamp with Timestamp
    # Ensure 'End Date' is not NaT, then check if it falls within the next 7 days, and is not completed
    due_soon = df[(df['End Date'].notna()) & 
                  (df['End Date'] >= pd.Timestamp(today)) & 
                  (df['End Date'] <= pd.Timestamp(week_ahead)) & 
                  (~df['Status'].str.contains('completed', case=False))]
    
    if not due_soon.empty:
        st.markdown(f'<div class="alert-warning"><strong>⏰ {len(due_soon)} Items Due This Week</strong></div>', unsafe_allow_html=True)
        with st.expander("View Items Due Soon"):
            for _, item in due_soon.iterrows():
                # Ensure item['End Date'] is a datetime object before calculating days_left
                if pd.notna(item['End Date']):
                    days_left = (item['End Date'].date() - today).days
                    st.markdown(f"• **{item['Name']}** (ID: {item['ID']}) - Due in {days_left} day(s)")
                else:
                    st.markdown(f"• **{item['Name']}** (ID: {item['ID']}) - Due soon (date N/A)")


    # Recently Completed Items (Fixed: comparing Timestamp with Timestamp for 'Delivered Date')
    month_ago = today - timedelta(days=30)
    recent_completed = df[(df['Delivered Date'].notna()) & 
                          (df['Delivered Date'] >= pd.Timestamp(month_ago)) & 
                          (df['Status'].str.contains('completed', case=False))] # Use case=False for flexible matching
    
    if not recent_completed.empty:
        st.markdown(f'<div class="alert-success"><strong>🎉 {len(recent_completed)} Recently Completed</strong></div>', unsafe_allow_html=True)
        with st.expander("View Recently Completed"):
            for _, item in recent_completed.iterrows():
                completed_date = item['Delivered Date'].strftime('%Y-%m-%d') if pd.notna(item['Delivered Date']) else 'N/A'
                st.markdown(f"• **{item['Name']}** (ID: {item['ID']}) - Completed: {completed_date}")
    else:
        st.info("No recent alerts to show at this moment.")


def show_footer():
    """Displays a simple footer with dashboard update time."""
    current_time = datetime.now()
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.8em;'>"
                f"Dashboard updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"PM Dashboard v4.2</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    show_footer()
