"""
Job Opportunities Dashboard
================================
Run with: streamlit run /Users/rioodwyer/PythonProjects-3/Over_Employed_Project/main.py

A clean dashboard for reviewing job opportunities one-by-one.
Use the sidebar filters to narrow down listings, select a job
from the dropdown, then Accept or Reject it.
"""

import streamlit as st
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
# Sets the browser tab title, icon, and layout width
st.set_page_config(
    page_title="Job Opportunities",
    page_icon="💼",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
# Tweak colours and card styles here
st.markdown("""
<style>
    .job-card {
        background: #1c1f2e;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 1rem;
    }
    .job-title  { font-size: 1.5rem; font-weight: 700; color: #e8eaf6; }
    .job-meta   { font-size: 0.85rem; color: #8b8fa8; margin-top: 4px; }
    .badge {
        display: inline-block;
        background: #0d1f3c; border: 1px solid #1e3a5f;
        color: #7eb3f7; border-radius: 20px;
        padding: 2px 10px; font-size: 0.75rem; margin-right: 6px;
    }
    .section-header {
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.15em; text-transform: uppercase;
        color: #5c6bc0; margin-bottom: 0.5rem; margin-top: 1.5rem;
    }
    hr { border-color: #2e3250 !important; }
    [data-testid="stSidebar"] { background: #12151f; border-right: 1px solid #2e3250; }
</style>
""", unsafe_allow_html=True)


# ── Sample Data ────────────────────────────────────────────────────────────────
# 🔧 REPLACE THIS with a real data source:
#    e.g. pd.read_csv("jobs.csv")  /  API call  /  database query
# Define required columns for the app
required_columns = [
    "title",
    "company",
    "location",
    "salary",
    "type",
    "status",
    "description",
]

starter_jobs = []

# Create DataFrame
df = pd.DataFrame(starter_jobs)

# Failsafe: ensure all required columns exist even if empty
if df.empty:
    df = pd.DataFrame(columns=required_columns)
else:
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""


# ── Sidebar Filters ────────────────────────────────────────────────────────────
# Everything in this block appears in the left sidebar panel
with st.sidebar:
    st.markdown("## Filters")
    st.divider()

    # Filter: Job type
    # Add/remove types
    all_types = ["All"] + sorted(df["type"].unique().tolist())
    selected_type = st.selectbox("Job Type", all_types)

    # Filter: Location keyword search
    # Simple text match against the location field
    location_search = st.text_input("Location contains", placeholder="e.g. Remote, New York")

    # Filter: Status
    # Add more status options to match your workflow (e.g. Accepted, Rejected)
    all_statuses = ["All"] + sorted(df["status"].unique().tolist())
    selected_status = st.selectbox("Status", all_statuses)

    st.divider()

    # Stats summary in sidebar
    st.markdown("### Summary")
    st.metric("Total Listings", len(df))
    st.metric("New",            len(df[df["status"] == "New"]))
    st.metric("Reviewed",       len(df[df["status"] == "Reviewed"]))


# ── Apply Filters ──────────────────────────────────────────────────────────────
# Narrows the DataFrame down based on whatever is selected in the sidebar
filtered_df = df.copy()

if selected_type != "All":
    filtered_df = filtered_df[filtered_df["type"] == selected_type]

if location_search:
    filtered_df = filtered_df[
        filtered_df["location"].str.contains(location_search, case=False, na=False)
    ]

if selected_status != "All":
    filtered_df = filtered_df[filtered_df["status"] == selected_status]


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Job Opportunities")
st.caption(f"Showing {len(filtered_df)} of {len(df)} listings")
st.divider()


# ── Handle empty results ───────────────────────────────────────────────────────
# If filters return nothing, stop rendering and show a message
if filtered_df.empty:
    st.warning("No jobs match your current filters. Try adjusting the sidebar.")
    st.stop()


# ── Job Selector Dropdown ──────────────────────────────────────────────────────
# Builds readable labels like "Senior Python Developer — Acme Corp (Remote)"
dropdown_labels = [
    f"{row['title']}  —  {row['company']}  ({row['location']})"
    for _, row in filtered_df.iterrows()
]

st.markdown('<p class="section-header">Select a Job to Review</p>', unsafe_allow_html=True)

# This is the main dropdown. Selecting an item here updates the detail card below.
selected_label = st.selectbox(
    label="Job listing",
    options=dropdown_labels,
    label_visibility="collapsed",  # hides the label since the header above acts as one
)

# Map the selected label back to a row in the filtered DataFrame
selected_index = dropdown_labels.index(selected_label)
job = filtered_df.iloc[selected_index]


# ── Job Detail Card ────────────────────────────────────────────────────────────
# Shows the full details of whichever job is selected in the dropdown above
st.markdown('<p class="section-header">Job Details</p>', unsafe_allow_html=True)

st.markdown(f"""
<div class="job-card">
    <div class="job-title">{job['title']}</div>
    <div class="job-meta">{job['company']} &nbsp;&middot;&nbsp; {job['location']} &nbsp;&middot;&nbsp; {job['salary']}</div>
    <br>
    <span class="badge">{job['type']}</span>
    <span class="badge">{job['status']}</span>
    <br><br>
    <p style="color:#c5c8e8; line-height:1.7">{job['description']}</p>
</div>
""", unsafe_allow_html=True)


# ── Accept / Reject Buttons ────────────────────────────────────────────────────
# Two side-by-side buttons. col_spacer pushes them to the left so they don't stretch.
st.markdown('<p class="section-header">Your Decision</p>', unsafe_allow_html=True)

col_accept, col_reject, col_spacer = st.columns([1, 1, 5])

with col_accept:
    accept_clicked = st.button(
        "Accept",
        use_container_width=True,
        type="primary",
    )

with col_reject:
    reject_clicked = st.button(
        "Reject",
        use_container_width=True,
        type="secondary",
    )


# ── Button Event Handlers ──────────────────────────────────────────────────────
# These blocks run whenever the corresponding button is clicked.
# `job` holds all fields for the currently selected listing:
#   job['title'], job['company'], job['location'], job['salary'], job['description']

if accept_clicked:
    # ACCEPT EVENT
    # ─────────────────────────────────────────────────────────────────────────
    # This code runs when the user clicks Accept.
    # Some ideas for what to do here:
    #   - Write the job to an "accepted" CSV or database table
    #   - Send yourself an email or Slack message
    #   - Call an external API to flag or bookmark the listing
    #   - Open the application link in a new browser tab
    #   - Update the job's status field to "Accepted"
    # ─────────────────────────────────────────────────────────────────────────
    st.success(f"Accepted: **{job['title']}** at {job['company']}")
    # TODO: add your accept logic here


if reject_clicked:
    # REJECT EVENT
    # ─────────────────────────────────────────────────────────────────────────
    # This code runs when the user clicks Reject.
    # Some ideas for what to do here:
    #   - Write the job to a "rejected" or blacklist CSV
    #   - Log the rejection with an optional reason (add a text_input above for that)
    #   - Update the job's status field to "Rejected" so it filters out next time
    #   - Send an automated decline response if you have contact info
    # ─────────────────────────────────────────────────────────────────────────
    st.error(f"Rejected: **{job['title']}** at {job['company']}")
    # TODO: add your reject logic here


# ── Quick-browse Table ─────────────────────────────────────────────────────────
# Shows all currently filtered jobs in a compact table at the bottom.
# Hidden by default to keep the page clean — click to expand.
st.divider()
st.markdown('<p class="section-header">All Filtered Listings</p>', unsafe_allow_html=True)

with st.expander("Show full list", expanded=False):
    # Change the column list here to show/hide fields in the table
    st.dataframe(
        filtered_df[["title", "company", "location", "salary", "type", "status"]],
        use_container_width=True,
        hide_index=True,
    )