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
from playwright_stealth import Stealth

from agent_implementation import run_url_agent
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Job Search",
    page_icon="",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background: #0a0b0e;
    color: #d4d8e8;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.search-panel {
    background: #0f1117;
    border: 1px solid #1e2235;
    border-radius: 4px;
    padding: 28px 32px 20px;
    margin-bottom: 2rem;
}

.field-label {
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a5080;
    margin-bottom: 6px;
}

.result-card {
    background: #0f1117;
    border: 1px solid #1e2235;
    border-left: 3px solid #3d5afe;
    border-radius: 2px;
    padding: 20px 24px;
    margin-bottom: 10px;
    transition: border-color 0.15s;
}

.result-card:hover { border-left-color: #7c8eff; }

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #e8eaf6;
    margin-bottom: 4px;
}

.card-meta {
    font-size: 0.75rem;
    color: #5a6080;
    margin-bottom: 10px;
}

.card-desc {
    font-size: 0.8rem;
    color: #9098b8;
    line-height: 1.6;
    margin-bottom: 10px;
}

.tag {
    display: inline-block;
    background: #13172a;
    border: 1px solid #1e2a4a;
    color: #5c7cff;
    border-radius: 2px;
    padding: 2px 8px;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    margin-right: 5px;
    margin-bottom: 4px;
}

.tag-salary { border-color: #1a3a2a; color: #4caf80; background: #0d1a12; }
.tag-type   { border-color: #2a1a3a; color: #9c70ff; background: #12091a; }

.result-count {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #3d4060;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

a.job-link {
    font-size: 0.72rem;
    color: #3d5afe;
    text-decoration: none;
    letter-spacing: 0.05em;
}

a.job-link:hover { color: #7c8eff; }

.source-badge {
    display: inline-block;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 1px 6px;
    border-radius: 2px;
    margin-left: 8px;
    vertical-align: middle;
}
.source-seek   { background: #0e2a1e; color: #2e8b57; border: 1px solid #1a4a2e; }
.source-indeed { background: #1a1a0e; color: #8b7a2e; border: 1px solid #3a3010; }

[data-testid="stSidebar"] { display: none; }

div[data-testid="stVerticalBlock"] > div:first-child { padding-top: 2rem; }

.stButton > button {
    background: #1a1f3a;
    border: 1px solid #2a3060;
    color: #7c8eff;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    border-radius: 2px;
    padding: 10px 28px;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #2a3060;
    border-color: #3d5afe;
    color: #fff;
}

.stTextInput > div > div > input,
.stMultiSelect > div,
.stSelectbox > div > div {
    background: #13172a !important;
    border-color: #1e2235 !important;
    color: #d4d8e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    border-radius: 2px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# Holds scraped results between Streamlit reruns.
# ══════════════════════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = []    # list of job dicts
if "searched" not in st.session_state:
    st.session_state.searched = False


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 style="font-size:2rem; letter-spacing:-0.02em; margin-bottom:0.2rem;">Job Search</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#3d4060; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:2rem;">SEEK + Indeed &nbsp;/&nbsp; Brisbane</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH FORM
# All input fields live here. Values are read when the Search button is clicked.
# ══════════════════════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="search-panel">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])
    col3, col4 = st.columns([2, 2])
    col5, col6 = st.columns([2, 2])

    with col1:
        st.markdown('<p class="field-label">Job Title / Keywords</p>', unsafe_allow_html=True)
        # ── AGENT HOOK: job_title ──────────────────────────────────────────────
        # Pass this value as the `q` param in your SEEK / Indeed URL.
        # SEEK:   https://www.seek.com.au/{job_title}-jobs/in-brisbane
        #         (replace spaces with dashes, e.g. "gym receptionist" -> "gym-receptionist-jobs")
        # Indeed: https://au.indeed.com/jobs?q={job_title}&l=Brisbane+QLD
        job_title = st.text_input("", placeholder="e.g. gym receptionist, sales assistant", key="job_title", label_visibility="collapsed")

    with col2:
        st.markdown('<p class="field-label">Industry / Field</p>', unsafe_allow_html=True)
        # ── AGENT HOOK: field ─────────────────────────────────────────────────
        # Used to narrow the SEEK classification.
        # SEEK URL param: &classification=<id>  (lookup classification IDs at seek.com.au)
        # For free-text: append to the job_title search query, e.g. "gym receptionist fitness"
        # Example classification IDs: Sport & Recreation = 6162, Retail = 6246
        field = st.text_input("", placeholder="e.g. fitness, retail, hospitality", key="field", label_visibility="collapsed")

    with col3:
        st.markdown('<p class="field-label">Example Companies</p>', unsafe_allow_html=True)
        # ── AGENT HOOK: companies ─────────────────────────────────────────────
        # Not directly usable in SEEK/Indeed URL params.
        # Use post-scrape filtering: filter results where job['company'] contains
        # any of these company names (case-insensitive partial match).
        # Example: [j for j in results if any(c.lower() in j['company'].lower() for c in companies)]
        companies_raw = st.text_input("", placeholder="e.g. Fitness Cartel, Goodlife", key="companies", label_visibility="collapsed")

    with col4:
        st.markdown('<p class="field-label">Work Type</p>', unsafe_allow_html=True)
        # ── AGENT HOOK: work_types ────────────────────────────────────────────
        # SEEK URL param: &worktype=<id>
        #   Full time = 242, Part time = 243, Casual/Vacation = 244, Contract = 245
        # Indeed URL param: &jt=<type>
        #   fulltime, parttime, contract, temporary, internship
        # Use post-scrape filtering as fallback if multi-select is used.
        work_types = st.multiselect(
            "",
            options=["Full time", "Part time", "Casual", "Contract"],
            key="work_types",
            label_visibility="collapsed"
        )
        final_types = ", ".join(work_types)



    with col5:
        st.markdown('<p class="field-label">Pay Range (AUD/hr or salary)</p>', unsafe_allow_html=True)
        # ── AGENT HOOK: pay_range ─────────────────────────────────────────────
        # SEEK URL param: &salaryrange=<min>-<max>&salarytype=annual  (or hourly)
        # Example: &salaryrange=50000-80000&salarytype=annual
        # For hourly: &salaryrange=20-35&salarytype=hourly
        # Post-scrape: filter results where job['salary'] exists and is in range.
        # Note: many listings don't show salary, so URL filtering is incomplete.
        pay_range = st.text_input("", placeholder="e.g. $25-$35/hr  or  $55k-$70k", key="pay_range", label_visibility="collapsed")

    with col6:
        st.markdown('<p class="field-label">Location</p>', unsafe_allow_html=True)
        # ── AGENT HOOK: location ──────────────────────────────────────────────
        # Pass directly into the SEEK and Indeed location fields.
        # SEEK:   append "in-{location}" to URL path, e.g. /gym-jobs/in-brisbane
        #         For suburb-level: use &where={suburb} param
        # Indeed: &l={location} param, e.g. &l=Moorooka+QLD
        # Drive-time filtering is not available via URL — post-scrape only.
        # To filter by drive time you would need the Google Maps Distance Matrix API:
        #   for each job, geocode the location then call the API with origin=Moorooka.
        location = st.text_input("", placeholder="e.g. within 20 min of Moorooka", key="location", label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Search button ──────────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([4, 1, 4])
    with btn_col:
        search_clicked = st.button("SEARCH", use_container_width=True)
    
    st.session_state.results = []


# ══════════════════════════════════════════════════════════════════════════════
# AGENT WORKFLOW — triggered when Search is clicked
# ══════════════════════════════════════════════════════════════════════════════
if search_clicked:
    st.session_state.searched = True
    st.session_state.agent_input = job_title + " " + field + " " + companies_raw + " " + final_types + " " + pay_range + " " + location
    print(st.session_state.agent_input)
    st.session_state.results = run_url_agent(st.session_state.agent_input)

    # ── STEP 1: Build URLs ─────────────────────────────────────────────────────
    # Construct SEEK and Indeed search URLs from the form inputs above.
    #
    # SEEK URL format:
    #   https://www.seek.com.au/{slug}-jobs/in-{location}?worktype={wt}&salaryrange={pay}
    #
    # SEEK slug = job_title with spaces replaced by dashes, lowercased
    # Example: "gym receptionist" -> "https://www.seek.com.au/gym-receptionist-jobs/in-brisbane"
    #
    # TODO: Build SEEK URL here
    #   seek_url = f"https://www.seek.com.au/{job_title.replace(' ', '-').lower()}-jobs/in-brisbane"
    #
    # TODO: Build Indeed URL here
    #   indeed_url = f"https://au.indeed.com/jobs?q={job_title.replace(' ', '+')}&l={location.replace(' ', '+')}"
    #
    # ── STEP 2: Scrape ────────────────────────────────────────────────────────
    # Call your scraper functions with the URLs built above.
    # Both return a list of dicts with keys: title, url, company, location,
    # job_type, description, salary (Indeed only), bullet_points (SEEK only), etc.
    #
    # from seek_scraper import get_seek_listings
    # from indeed_scraper import get_indeed_listings
    #
    # TODO: Call scrapers and tag each result with its source
    #   seek_results   = [dict(source="SEEK",   **j) for j in get_seek_listings(seek_url)]
    #   indeed_results = [dict(source="Indeed", **j) for j in get_indeed_listings(indeed_url)]
    #   raw_results    = seek_results + indeed_results
    #
    # ── STEP 3: Post-scrape filtering ─────────────────────────────────────────
    # Apply filters that can't be done via URL params.
    #
    # Company filter (if companies_raw is filled in):
    #   companies = [c.strip() for c in companies_raw.split(",") if c.strip()]
    #   if companies:
    #       raw_results = [j for j in raw_results
    #                      if any(c.lower() in j.get("company","").lower() for c in companies)]
    #
    # Work type filter:
    #   if work_types:
    #       raw_results = [j for j in raw_results
    #                      if any(wt.lower() in (j.get("job_type") or "").lower() for wt in work_types)]
    #
    # ── STEP 4: Store results ─────────────────────────────────────────────────
    # Save to session state so results persist across Streamlit reruns.
    #
    #   st.session_state.results = raw_results
    #
    # ── PLACEHOLDER (remove once scrapers are wired in) ───────────────────────
    # replace with raw_results above


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.searched:
    results = st.session_state.results

    st.markdown(
        f'<p class="result-count">{len(results)} results found</p>',
        unsafe_allow_html=True
    )

    if not results:
        st.markdown(
            '<p style="color:#3d4060; font-size:0.85rem;">No results yet. Wire up the scrapers in the AGENT WORKFLOW section above.</p>',
            unsafe_allow_html=True
        )
    else:
        for job in results:
            source      = job.get("source", "")
            title       = job.get("title", "Untitled")
            company     = job.get("company", "")
            location    = job.get("location", "")
            job_type    = job.get("job_type", "")
            salary      = job.get("salary", "")
            description = job.get("description", "")
            url         = job.get("url", "#")
            date_posted = job.get("date_posted", "")
            bullets     = job.get("bullet_points") or []
            perks       = job.get("perks") or []

            # Source badge HTML
            badge_class = "source-seek" if source == "SEEK" else "source-indeed"
            source_html = f'<span class="source-badge {badge_class}">{source}</span>' if source else ""

            # Tags row
            tags_html = ""
            if salary:
                tags_html += f'<span class="tag tag-salary">{salary}</span>'
            if job_type:
                tags_html += f'<span class="tag tag-type">{job_type}</span>'
            for perk in (perks[:3] if perks else []):
                tags_html += f'<span class="tag">{perk}</span>'
            for b in (bullets[:2] if bullets else []):
                tags_html += f'<span class="tag">{b[:60]}</span>'

            st.markdown(f"""
<div class="result-card">
    <div class="card-title">{title}{source_html}</div>
    <div class="card-meta">{company} &nbsp;/&nbsp; {location} &nbsp;/&nbsp; {date_posted}</div>
    <div style="margin-bottom:10px">{tags_html}</div>
    <div class="card-desc">{description}</div>
    <a class="job-link" href="{url}" target="_blank">VIEW LISTING &rarr;</a>
</div>
""", unsafe_allow_html=True)