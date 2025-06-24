import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from typing import Dict, List, Tuple, Optional
import numpy as np

# --- Configuration ---
TIMEZONE_OPTIONS = {
    'UTC': 'UTC',
    'US/Eastern': 'America/New_York',
    'US/Central': 'America/Chicago',
    'US/Mountain': 'America/Denver',
    'US/Pacific': 'America/Los_Angeles',
    'Europe/London': 'Europe/London',
    'Europe/Paris': 'Europe/Paris',
    'Asia/Tokyo': 'Asia/Tokyo',
    'Asia/Shanghai': 'Asia/Shanghai',
    'Asia/Taipei': 'Asia/Taipei',
    'Australia/Sydney': 'Australia/Sydney'
}

# --- Page Configuration ---
st.set_page_config(
    page_title="Enhanced PM Dashboard",
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
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .alert-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-color: #f1aeb5;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .project-hierarchy {
        font-family: 'Courier New', monospace;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
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

        return 1 + self.get_project_depth(parent_id)

    def get_sequence_order(self) -> pd.DataFrame:
        """Return DataFrame with sequence ordering"""
        df_with_sequence = self.df.copy()

        # Add depth information
        df_with_sequence['Hierarchy_Depth'] = df_with_sequence['ID'].apply(self.get_project_depth)

        # Create type priority mapping
        type_priority = {
            'program': 1,
            'project': 2,
            'sub-project': 3,
            'task': 4,
            'milestone': 5
        }

        df_with_sequence['Type_Priority'] = df_with_sequence['Type'].str.lower().map(type_priority).fillna(99)

        # Sort by multiple criteria for proper sequencing
        df_with_sequence = df_with_sequence.sort_values([
            'Hierarchy_Depth',
            'Type_Priority',
            'Start Date',
            'Name'
        ], na_position='last')

        return df_with_sequence

class TimezoneManager:
    """Handles timezone conversions and display"""

    def __init__(self, timezone_name: str = 'UTC'):
        self.timezone = pytz.timezone(timezone_name)
        self.timezone_name = timezone_name

    def localize_datetime(self, dt: datetime) -> datetime:
        """Convert datetime to specified timezone"""
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(self.timezone)

    def format_datetime(self, dt: datetime, format_str: str = '%Y-%m-%d %H:%M %Z') -> str:
        """Format datetime with timezone info"""
        if pd.isna(dt):
            return 'N/A'
        localized_dt = self.localize_datetime(dt)
        return localized_dt.strftime(format_str)

    def get_current_time(self) -> datetime:
        """Get current time in specified timezone"""
        return datetime.now(self.timezone)

class DataProcessor:
    """Handles data cleaning and preprocessing"""

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the uploaded DataFrame"""
        df_clean = df.copy()

        # Clean date columns
        date_cols = ['Start Date', 'End Date', 'Delivered Date']
        for col in date_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

        # Clean ID columns
        df_clean['ID'] = df_clean['ID'].astype(str).str.strip()
        if 'Parent ID' in df_clean.columns:
            df_clean['Parent ID'] = df_clean['Parent ID'].astype(str).str.strip().fillna('')
        else:
            df_clean['Parent ID'] = ''

        # Clean text columns
        for col in ['Name', 'Type']:
            if col not in df_clean.columns:
                raise ValueError(f"Missing required column: {col}")
            df_clean[col] = df_clean[col].astype(str).fillna('')

        # Handle Progress column
        if 'Progress' not in df_clean.columns:
            df_clean['Progress'] = df_clean['Delivered Date'].apply(
                lambda x: 100 if pd.notna(x) else 0
            )
        else:
            df_clean['Progress'] = pd.to_numeric(df_clean['Progress'], errors='coerce').fillna(0)
            df_clean['Progress'] = df_clean['Progress'].clip(0, 100)  # Ensure 0-100 range

        # Handle Status column
        if 'Status' not in df_clean.columns:
            df_clean['Status'] = 'N/A'
        else:
            df_clean['Status'] = df_clean['Status'].astype(str).str.lower().fillna('n/a')

        return df_clean

    @staticmethod
    def infer_status(df: pd.DataFrame, timezone_manager: TimezoneManager) -> pd.DataFrame:
        """Infer project status based on dates and current time"""
        df_status = df.copy()
        today = timezone_manager.get_current_time().replace(tzinfo=None).date()

        def calculate_status(row):
            current_status = str(row.get('Status', 'n/a')).lower()

            # Completed cases
            if pd.notna(row['Delivered Date']):
                if pd.notna(row['End Date']) and row['Delivered Date'].date() <= row['End Date'].date():
                    return 'completed'
                elif pd.notna(row['End Date']) and row['Delivered Date'].date() > row['End Date'].date():
                    return 'completed (late)'
                else:
                    return 'completed'

            # Overdue case
            if pd.notna(row['End Date']) and row['End Date'].date() < today:
                return 'overdue'

            # In progress case
            if (pd.notna(row['Start Date']) and pd.notna(row['End Date']) and
                    row['Start Date'].date() <= today <= row['End Date'].date()):
                return 'in progress'

            # Not started case
            if pd.notna(row['Start Date']) and row['Start Date'].date() > today:
                return 'not started'

            return current_status if current_status != 'n/a' else 'not started'

        df_status['Status'] = df_status.apply(calculate_status, axis=1)
        return df_status

class DashboardVisualizer:
    """Handles dashboard visualizations"""

    def __init__(self, timezone_manager: TimezoneManager):
        self.timezone_manager = timezone_manager

    def create_enhanced_gantt(self, df: pd.DataFrame, date_range: Tuple[datetime, datetime]) -> go.Figure:
        """
        Create an enhanced Gantt chart with a scalable timeline and range slider.
        """
        if df.empty:
            return go.Figure()

        # Prepare data based on the date range from the Streamlit widget
        gantt_df = df[
            (df['Start Date'] <= pd.to_datetime(date_range[1])) &
            (df['End Date'] >= pd.to_datetime(date_range[0]))
        ].copy()

        if gantt_df.empty:
            st.info("No project items fall within the selected date range.")
            return go.Figure()

        # Handle milestones for better visibility
        milestone_mask = (gantt_df['Type'].str.lower() == 'milestone')
        gantt_df.loc[milestone_mask, 'End Date'] = gantt_df.loc[milestone_mask, 'Start Date'] + timedelta(days=1)

        # Sort for proper hierarchical display on the y-axis
        sequencer = ProjectSequencer(gantt_df)
        gantt_df = sequencer.get_sequence_order()

        # Create figure
        fig = go.Figure()

        # Color mapping
        color_map = {
            'Program': '#1f77b4',
            'Project': '#ff7f0e',
            'Sub-Project': '#2ca02c',
            'Task': '#d62728',
            'Milestone': '#9467bd'
        }

        # Add bars for each project
        # Note: We iterate through sorted df to add traces one by one, preserving order.
        for index, row in gantt_df.iterrows():
            fig.add_trace(go.Bar(
                name=row['Type'],
                y=[row['Name']],
                x=[(row['End Date'] - row['Start Date']).total_seconds() * 1000], # Duration in ms
                base=[row['Start Date']],
                orientation='h',
                marker_color=color_map.get(row['Type'], '#17becf'),
                text=f"{row['Progress']:.0f}%",
                textposition='inside',
                hovertemplate=(
                    f"<b>{row['Name']}</b><br>"
                    f"Type: {row['Type']}<br>"
                    "Start: %{base|%Y-%m-%d}<br>"
                    "End: %{x|%Y-%m-%d}<br>" # Custom formatting needed as x is duration
                    f"Progress: {row['Progress']:.0f}%<br>"
                    f"Status: {row['Status'].title()}<extra></extra>"
                ),
                legendgroup=row['Type'],
                showlegend=(row['Type'] not in [t.name for t in fig.data]) # Show legend item only once per type
            ))

        # Update layout with scalable timeline controls
        fig.update_layout(
            title_text='Enhanced Project Timeline with Scalable View',
            yaxis_title='Projects',
            height=max(600, len(gantt_df) * 25),
            barmode='overlay',
            hovermode='closest',
            legend=dict(title="Item Type"),
            xaxis=dict(
                title='Timeline',
                type='date',
                # Add range selector buttons for scalable views
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="Week", step="day", stepmode="backward"),
                        dict(count=1, label="Month", step="month", stepmode="backward"),
                        dict(count=3, label="Quarter", step="month", stepmode="backward"),
                        dict(count=6, label="Half-Year", step="month", stepmode="backward"),
                        dict(count=1, label="Year", step="year", stepmode="todate"),
                        dict(step="all", label="All")
                    ])
                ),
                # Add the range slider for manual adjustments
                rangeslider=dict(
                    visible=True
                ),
            )
        )
        
        # Reverse y-axis to show parent projects on top
        fig.update_yaxes(autorange="reversed")

        return fig

    def create_status_summary_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a status summary chart"""
        status_counts = df['Status'].value_counts()

        colors = {
            'completed': '#28a745',
            'completed (late)': '#20c997',
            'in progress': '#007bff',
            'not started': '#6c757d',
            'overdue': '#dc3545',
            'on hold': '#ffc107'
        }

        fig = go.Figure(data=[
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                marker_colors=[colors.get(status.lower(), '#17a2b8') for status in status_counts.index],
                textinfo='label+percent',
                textposition='auto',
            )
        ])

        fig.update_layout(
            title='Project Status Distribution',
            height=400
        )

        return fig

# --- Main Application ---
def main():
    # Header
    st.markdown('<div class="main-header"><h1>📊 Enhanced Project Management Dashboard</h1></div>',
                unsafe_allow_html=True)

    # Sidebar for timezone selection
    st.sidebar.header("⚙️ Settings")
    selected_timezone = st.sidebar.selectbox(
        "Select Timezone:",
        options=list(TIMEZONE_OPTIONS.keys()),
        index=list(TIMEZONE_OPTIONS.keys()).index('UTC')
    )

    timezone_manager = TimezoneManager(TIMEZONE_OPTIONS[selected_timezone])

    # Display current time
    current_time = timezone_manager.get_current_time()
    st.sidebar.info(f"Current time ({selected_timezone}): {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # File upload section
    st.markdown("### 📁 Upload Project Data")
    uploaded_file = st.file_uploader(
        "Choose an Excel file (.xlsx or .xls)",
        type=["xlsx", "xls"],
        help="Upload your project management Excel file with columns: ID, Name, Type, Start Date, End Date"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Processing your file..."):
                # Read and process data
                df = pd.read_excel(uploaded_file, engine='openpyxl')
                df_clean = DataProcessor.clean_dataframe(df)
                df_processed = DataProcessor.infer_status(df_clean, timezone_manager)

            st.success("✅ File successfully processed!")
            st.session_state.df = df_processed
            st.session_state.timezone_manager = timezone_manager

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your Excel file contains the required columns: ID, Name, Type, Start Date, End Date")
            return

    # Main dashboard
    if 'df' in st.session_state:
        df = st.session_state.df
        timezone_manager = st.session_state.timezone_manager

        # Sidebar filters
        st.sidebar.header("🔍 Filters")

        # Project/Program filter
        top_level_projects = df[df['Type'].str.lower().isin(['project', 'program'])]['Name'].unique()
        selected_projects = st.sidebar.multiselect(
            "Filter by Project/Program:",
            top_level_projects,
            help="Select specific projects to focus on"
        )

        # Type filter
        all_types = sorted(df['Type'].unique())
        selected_types = st.sidebar.multiselect(
            "Filter by Type:",
            all_types,
            default=all_types
        )

        # Status filter
        all_statuses = sorted(df['Status'].unique())
        selected_statuses = st.sidebar.multiselect(
            "Filter by Status:",
            all_statuses,
            default=all_statuses
        )

        # Apply filters
        filtered_df = df.copy()

        if selected_projects:
            # Get hierarchical children
            selected_ids = df[df['Name'].isin(selected_projects)]['ID'].tolist()

            # Find all descendants
            all_included_ids = set(selected_ids)
            current_ids = set(selected_ids)

            while current_ids:
                children = df[df['Parent ID'].isin(current_ids)]['ID'].tolist()
                new_children = set(children) - all_included_ids
                if not new_children:
                    break
                all_included_ids.update(new_children)
                current_ids = new_children

            filtered_df = df[df['ID'].isin(all_included_ids)]

        filtered_df = filtered_df[
            filtered_df['Type'].isin(selected_types) &
            filtered_df['Status'].isin(selected_statuses)
        ]

        # Navigation
        st.sidebar.header("📋 Navigation")
        tab = st.sidebar.radio(
            "Select View:",
            ["📊 Dashboard", "📈 Analytics", "📋 Data Table", "🏗️ Project Hierarchy"]
        )

        if tab == "📊 Dashboard":
            show_dashboard(filtered_df, timezone_manager)
        elif tab == "📈 Analytics":
            show_analytics(filtered_df, timezone_manager)
        elif tab == "📋 Data Table":
            show_data_table(filtered_df, timezone_manager)
        elif tab == "🏗️ Project Hierarchy":
            show_project_hierarchy(filtered_df, timezone_manager)

def show_dashboard(df: pd.DataFrame, timezone_manager: TimezoneManager):
    """Main dashboard view"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("📅 Project Timeline")

        # Date range selector to filter data BEFORE rendering the chart
        if not df.empty and 'Start Date' in df.columns and 'End Date' in df.columns:
            min_date = df['Start Date'].min().date() if pd.notna(df['Start Date'].min()) else datetime.now().date()
            max_date = df['End Date'].max().date() if pd.notna(df['End Date'].max()) else datetime.now().date() + timedelta(days=365)

            date_range = st.date_input(
                "Select a master date range to filter items:",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date,
                help="This filters the items shown in the Gantt chart below. Use the chart's internal controls to zoom."
            )

            if len(date_range) == 2:
                with st.spinner("Generating Gantt chart..."):
                    visualizer = DashboardVisualizer(timezone_manager)
                    gantt_fig = visualizer.create_enhanced_gantt(df, date_range)
                    st.plotly_chart(gantt_fig, use_container_width=True)
            else:
                st.info("Please select both a start and end date for the master filter.")

    with col2:
        st.header("📊 Quick Stats")

        # KPI metrics
        total_items = len(df)
        completed_items = len(df[df['Status'].str.contains('completed', case=False, na=False)])
        overdue_items = len(df[df['Status'] == 'overdue'])
        in_progress_items = len(df[df['Status'] == 'in progress'])

        st.metric("Total Items", total_items)
        st.metric("Completed", completed_items, delta=f"{(completed_items/total_items*100):.1f}%" if total_items > 0 else "0%")
        st.metric("In Progress", in_progress_items)
        st.metric("Overdue", overdue_items, delta_color="inverse", delta=f"{overdue_items} item(s)")

        # Status chart
        if not df.empty:
            visualizer = DashboardVisualizer(timezone_manager)
            status_fig = visualizer.create_status_summary_chart(df)
            st.plotly_chart(status_fig, use_container_width=True)

        # Alerts
        show_alerts(df, timezone_manager)

def show_analytics(df: pd.DataFrame, timezone_manager: TimezoneManager):
    """Analytics view with detailed charts"""
    if df.empty:
        st.info("No data available for analytics based on current filters.")
        return

    st.header("📈 Project Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Progress distribution
        if 'Progress' in df.columns:
            fig_progress = px.histogram(
                df,
                x='Progress',
                nbins=20,
                title='Progress Distribution',
                labels={'Progress': 'Progress (%)', 'count': 'Number of Items'}
            )
            st.plotly_chart(fig_progress, use_container_width=True)

    with col2:
        # Type distribution
        type_counts = df['Type'].value_counts()
        fig_types = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title='Items by Type',
            labels={'x': 'Type', 'y': 'Count'}
        )
        st.plotly_chart(fig_types, use_container_width=True)

    # Timeline analysis
    if 'Start Date' in df.columns and 'End Date' in df.columns:
        df_timeline = df.dropna(subset=['Start Date', 'End Date']).copy()
        if not df_timeline.empty:
            df_timeline['Duration'] = (df_timeline['End Date'] - df_timeline['Start Date']).dt.days

            fig_duration = px.scatter(
                df_timeline,
                x='Start Date',
                y='Duration',
                color='Type',
                size='Progress',
                hover_data=['Name', 'Status'],
                title='Project Duration vs. Start Date'
            )
            st.plotly_chart(fig_duration, use_container_width=True)

def show_data_table(df: pd.DataFrame, timezone_manager: TimezoneManager):
    """Data table view"""
    st.header("📋 Project Data Table")

    if df.empty:
        st.info("No data to display based on current filters.")
        return

    # Add export functionality
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"project_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Progress": st.column_config.ProgressColumn(
                "Progress",
                help="Project completion percentage",
                format="%.0f%%",
                min_value=0,
                max_value=100,
            ),
             "Start Date": st.column_config.DateColumn("Start Date", format="YYYY-MM-DD"),
             "End Date": st.column_config.DateColumn("End Date", format="YYYY-MM-DD"),
             "Delivered Date": st.column_config.DateColumn("Delivered Date", format="YYYY-MM-DD"),
        }
    )

def show_project_hierarchy(df: pd.DataFrame, timezone_manager: TimezoneManager):
    """Project hierarchy view"""
    st.header("🏗️ Project Hierarchy")

    if df.empty:
        st.info("No data to display based on current filters.")
        return

    sequencer = ProjectSequencer(df)
    full_sorted_df = sequencer.get_sequence_order()
    
    # Filter the sorted df to only the items that should be displayed
    df_to_display = full_sorted_df[full_sorted_df['ID'].isin(df['ID'])]

    # Get top-level items from the filtered set
    top_level = df_to_display[
        df_to_display['Parent ID'].isin(['', 'nan']) | 
        (~df_to_display['Parent ID'].isin(df_to_display['ID']))
    ].copy()

    def display_project_tree(parent_df: pd.DataFrame, level: int = 0):
        """Recursively display project hierarchy"""
        for _, row in parent_df.iterrows():
            indent = "    " * level
            status_emoji = {
                'completed': '✅',
                'completed (late)': '✅',
                'in progress': '🔄',
                'overdue': '🔴',
                'not started': '⏳'
            }.get(row['Status'], '⚪')

            progress_bar = "█" * int(row['Progress'] // 10) + "░" * (10 - int(row['Progress'] // 10))

            st.markdown(f"""
            <div class="project-hierarchy">
                <pre style="margin: 0; padding: 0; white-space: pre-wrap;">{indent}{status_emoji} <strong>{row['Name']}</strong> ({row['Type']})</pre>
                <pre style="margin: 0; padding: 0; white-space: pre-wrap;">{indent}  Progress: {progress_bar} {row['Progress']:.0f}%</pre>
                <pre style="margin: 0; padding: 0; white-space: pre-wrap;">{indent}  Status: {row['Status'].title()}</pre>
                <pre style="margin: 0; padding: 0; white-space: pre-wrap;">{indent}  Duration: {row['Start Date'].strftime('%Y-%m-%d') if pd.notna(row['Start Date']) else 'N/A'} → {row['End Date'].strftime('%Y-%m-%d') if pd.notna(row['End Date']) else 'N/A'}</pre>
            </div>
            """, unsafe_allow_html=True)

            # Find and display children from the filtered & sorted dataframe
            children = df_to_display[df_to_display['Parent ID'] == row['ID']]
            if not children.empty:
                display_project_tree(children, level + 1)

    display_project_tree(top_level)

def show_alerts(df: pd.DataFrame, timezone_manager: TimezoneManager):
    """Show project alerts"""
    st.subheader("🚨 Alerts")

    today = timezone_manager.get_current_time().date()

    # Overdue items
    overdue = df[df['Status'] == 'overdue']
    if not overdue.empty:
        st.markdown(f'<div class="alert-danger"><strong>⚠️ {len(overdue)} Overdue Items</strong></div>',
                      unsafe_allow_html=True)
        with st.expander("View Overdue Items"):
            for _, item in overdue.iterrows():
                st.markdown(f"• **{item['Name']}** - Due: {item['End Date'].strftime('%Y-%m-%d') if pd.notna(item['End Date']) else 'N/A'}")

    # Due soon
    week_ahead = today + timedelta(days=7)
    due_soon = df[
        (df['End Date'].notna()) &
        (df['End Date'].dt.date <= week_ahead) &
        (df['End Date'].dt.date >= today) &
        (~df['Status'].str.contains('completed'))
    ]

    if not due_soon.empty:
        st.markdown(f'<div class="alert-warning"><strong>⏰ {len(due_soon)} Items Due This Week</strong></div>',
                      unsafe_allow_html=True)
        with st.expander("View Items Due Soon"):
            for _, item in due_soon.iterrows():
                days_left = (item['End Date'].date() - today).days
                st.markdown(f"• **{item['Name']}** - {days_left} day(s) left")

    # Recently completed
    month_ago = today - timedelta(days=30)
    recent_completed = df[
        (df['Delivered Date'].notna()) &
        (df['Delivered Date'].dt.date >= month_ago) &
        (df['Status'].str.contains('completed'))
    ]

    if not recent_completed.empty:
        st.markdown(f'<div class="alert-success"><strong>🎉 {len(recent_completed)} Recently Completed</strong></div>',
                      unsafe_allow_html=True)
        with st.expander("View Recently Completed"):
            for _, item in recent_completed.iterrows():
                st.markdown(f"• **{item['Name']}** - Completed: {item['Delivered Date'].strftime('%Y-%m-%d') if pd.notna(item['Delivered Date']) else 'N/A'}")

# --- Footer ---
def show_footer(timezone_manager: TimezoneManager):
    """Display footer with current time"""
    current_time = timezone_manager.get_current_time()
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.8em;'>"
                f"Dashboard updated: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
                f"Enhanced PM Dashboard v2.0</div>",
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    if 'timezone_manager' in st.session_state:
        show_footer(st.session_state.timezone_manager)
