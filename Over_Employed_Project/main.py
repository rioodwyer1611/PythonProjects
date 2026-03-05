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
    padding: 20px 24px 14px;
    margin-bottom: 4px;
    transition: border-color 0.15s;
}

.result-card:hover { border-left-color: #7c8eff; }
.result-card.accepted { border-left-color: #2e8b57 !important; background: #0a120d; }
.result-card.rejected { border-left-color: #8b2e2e !important; background: #120a0a; opacity: 0.5; }

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
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    border-radius: 2px;
    padding: 6px 16px;
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
# ══════════════════════════════════════════════════════════════════════════════
if "results"   not in st.session_state: st.session_state.results   = []
if "searched"  not in st.session_state: st.session_state.searched  = False
if "decisions" not in st.session_state: st.session_state.decisions = {}  # job_id -> "accepted" | "rejected"
if "accepted"  not in st.session_state: st.session_state.accepted  = []  # list of accepted job dicts
if "rejected"  not in st.session_state: st.session_state.rejected  = []  # list of rejected job dicts


def job_id(job: dict) -> str:
    """Stable unique key for a job — url if present, else title|company."""
    return job.get("url") or f"{job.get('title', '')}|{job.get('company', '')}"


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 style="font-size:2rem; letter-spacing:-0.02em; margin-bottom:0.2rem;">Job Search</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#3d4060; font-size:0.75rem; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:2rem;">SEEK + Indeed &nbsp;/&nbsp; Brisbane</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH FORM
# ══════════════════════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="search-panel">', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])
    col3, col4 = st.columns([2, 2])
    col5, col6 = st.columns([2, 2])

    with col1:
        st.markdown('<p class="field-label">Job Title / Keywords</p>', unsafe_allow_html=True)
        job_title = st.text_input("", placeholder="e.g. gym receptionist, sales assistant", key="job_title", label_visibility="collapsed")

    with col2:
        st.markdown('<p class="field-label">Industry / Field</p>', unsafe_allow_html=True)
        field = st.text_input("", placeholder="e.g. fitness, retail, hospitality", key="field", label_visibility="collapsed")

    with col3:
        st.markdown('<p class="field-label">Example Companies</p>', unsafe_allow_html=True)
        companies_raw = st.text_input("", placeholder="e.g. Fitness Cartel, Goodlife", key="companies", label_visibility="collapsed")

    with col4:
        st.markdown('<p class="field-label">Work Type</p>', unsafe_allow_html=True)
        work_types = st.multiselect(
            "",
            options=["Full time", "Part time", "Casual", "Contract"],
            key="work_types",
            label_visibility="collapsed"
        )
        final_types = ", ".join(work_types)

    with col5:
        st.markdown('<p class="field-label">Pay Range (AUD/hr or salary)</p>', unsafe_allow_html=True)
        pay_range = st.text_input("", placeholder="e.g. $25-$35/hr  or  $55k-$70k", key="pay_range", label_visibility="collapsed")

    with col6:
        st.markdown('<p class="field-label">Location</p>', unsafe_allow_html=True)
        location = st.text_input("", placeholder="e.g. within 20 min of Moorooka", key="location", label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([4, 1, 4])
    with btn_col:
        search_clicked = st.button("SEARCH", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT WORKFLOW — triggered when Search is clicked
# ══════════════════════════════════════════════════════════════════════════════
if search_clicked:
    st.session_state.searched  = True
    st.session_state.decisions = {}   # clear decisions on new search
    st.session_state.accepted  = []
    st.session_state.rejected  = []
    st.session_state.agent_input = job_title + " " + field + " " + companies_raw + " " + final_types + " " + pay_range + " " + location
    print(st.session_state.agent_input)
    st.session_state.results = run_url_agent(st.session_state.agent_input)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.searched:
    results = st.session_state.results
    rejected_ids = {job_id(j) for j in st.session_state.rejected}
    visible_results = [j for j in results if job_id(j) not in rejected_ids]

    n_accepted = sum(1 for v in st.session_state.decisions.values() if v == "accepted")
    n_rejected = sum(1 for v in st.session_state.decisions.values() if v == "rejected")
    n_pending  = len(results) - n_accepted - n_rejected

    st.markdown(
        f'<p class="result-count">'
        f'{len(results)} results &nbsp;&nbsp; {n_accepted} accepted &nbsp;&nbsp; {n_rejected} rejected &nbsp;&nbsp; {n_pending} pending'
        f'</p>',
        unsafe_allow_html=True
    )

    if not visible_results:
        st.markdown(
            '<p style="color:#3d4060; font-size:0.85rem;">No results yet. Wire up the scrapers in the AGENT WORKFLOW section above.</p>',
            unsafe_allow_html=True
        )
    else:
        for i, job in enumerate(visible_results):
            jid         = job_id(job)
            decision    = st.session_state.decisions.get(jid)
            source      = job.get("source", "")
            title       = job.get("title", "Untitled")
            company     = job.get("company", "")
            loc         = job.get("location", "")
            job_type    = job.get("job_type", "")
            salary      = job.get("salary", "")
            description = job.get("description", "")
            url         = job.get("url", "#")
            date_posted = job.get("date_posted", "")
            bullets     = job.get("bullet_points") or []
            perks       = job.get("perks") or []

            # Source badge
            badge_class = "source-seek" if source == "SEEK" else "source-indeed"
            source_html = f'<span class="source-badge {badge_class}">{source}</span>' if source else ""

            # Tags row
            tags_html = ""
            if salary:   tags_html += f'<span class="tag tag-salary">{salary}</span>'
            if job_type: tags_html += f'<span class="tag tag-type">{job_type}</span>'
            for perk in (perks[:3] if perks else []):   tags_html += f'<span class="tag">{perk}</span>'
            for b in (bullets[:2] if bullets else []):  tags_html += f'<span class="tag">{b[:60]}</span>'

            # Card style changes based on decision
            card_class = {
                "accepted": "result-card accepted",
                "rejected": "result-card rejected",
            }.get(decision, "result-card")

            # ── Job card HTML ──────────────────────────────────────────────────
            st.markdown(f"""
<div class="{card_class}">
    <div class="card-title">{title}{source_html}</div>
    <div class="card-meta">{company} &nbsp;/&nbsp; {loc} &nbsp;/&nbsp; {date_posted}</div>
    <div style="margin-bottom:10px">{tags_html}</div>
    <div class="card-desc">{description}</div>
    <a class="job-link" href="{url}" target="_blank">VIEW LISTING &rarr;</a>
</div>
""", unsafe_allow_html=True)

            # ── Accept / Reject buttons ────────────────────────────────────────
            btn_accept_col, btn_reject_col, _ = st.columns([1, 1, 8])

            with btn_accept_col:
                accept_label = "✓ ACCEPTED" if decision == "accepted" else "ACCEPT"
                if st.button(accept_label, key=f"accept_{i}", use_container_width=True):
                    # ── ACCEPT EVENT ──────────────────────────────────────────
                    # `job` dict is available here with all fields.
                    # TODO: add side-effects, e.g.:
                    #   pd.DataFrame([job]).to_csv("accepted_jobs.csv", mode="a", header=False, index=False)
                    st.session_state.decisions[jid] = "accepted"
                    if job not in st.session_state.accepted:
                        st.session_state.accepted.append(job)
                    st.rerun()

            with btn_reject_col:
                reject_label = "✗ REJECTED" if decision == "rejected" else "REJECT"
                if st.button(reject_label, key=f"reject_{i}", use_container_width=True):
                    # ── REJECT EVENT ──────────────────────────────────────────
                    # `job` dict is available here with all fields.
                    # TODO: add side-effects, e.g.:
                    #   pd.DataFrame([job]).to_csv("rejected_jobs.csv", mode="a", header=False, index=False)
                    st.session_state.decisions[jid] = "rejected"
                    if job not in st.session_state.rejected:
                        st.session_state.rejected.append(job)
                    st.rerun()

            st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)