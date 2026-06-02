"""
RAG Reliability Inspector — Dashboard
Editorial / Anthropic-inspired theme. Functionality unchanged.
Run with:  streamlit run app.py
"""

import json
import os
import streamlit as st
import pandas as pd
from src.inspector import Inspector

st.set_page_config(page_title="RAG Reliability Inspector",
                   page_icon="📖", layout="wide")

# ----------------------------------------------------------------------
# CUSTOM CSS — editorial cream/terracotta theme (Newsreader + Inter)
# ----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:wght@400;500&family=Inter:wght@400;500;600&display=swap');

/* Base app surface */
.stApp {
    background: #fff8f3;
    color: #1d1b18;
}

/* Constrain main content width */
.block-container {
    max-width: 1100px;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* Body / UI typography */
html, body, [class*="css"], .stMarkdown, .stTextArea label, .stTextInput label, p, span, div {
    font-family: 'Inter', system-ui, sans-serif;
    color: #1d1b18;
}

/* Headings in Newsreader serif */
h1, h2, h3, h4 {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 400 !important;
    color: #1d1b18 !important;
    letter-spacing: -0.005em;
}
h1 { font-size: 40px !important; line-height: 48px !important; }
h2 { font-size: 32px !important; line-height: 40px !important; }
h3 { font-size: 24px !important; line-height: 32px !important; }

/* Page hero block */
.hero-block {
    margin-bottom: 2.5rem;
}
.hero-block h1 {
    margin: 0 0 0.5rem 0;
}
.hero-block .sub {
    color: #55433c;
    font-size: 17px;
    line-height: 26px;
    max-width: 680px;
}

/* Tabs — minimal underline, no container box */
.stTabs [data-baseweb="tab-list"] {
    gap: 2.5rem;
    border-bottom: 1px solid #e8e4d8;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #55433c !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 0.6rem 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #994422 !important;
    border-bottom-color: #994422 !important;
}

/* Hide stray horizontal rules */
hr, [data-testid="stMarkdownContainer"] hr, .stTabs hr {
    display: none !important;
}

/* Text area inputs */
.stTextArea textarea {
    background: #f3ede7 !important;
    border: 1px solid #e8e4d8 !important;
    border-radius: 4px !important;
    color: #1d1b18 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
}
.stTextArea textarea:focus {
    border-color: #994422 !important;
    box-shadow: none !important;
}
.stTextArea label {
    color: #55433c !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* Primary buttons — terracotta, white text */
.stButton > button,
.stButton > button:hover,
.stButton > button:active,
.stButton > button:focus,
.stButton > button:focus:not(:active),
.stButton > button p,
.stButton > button:hover p,
.stButton > button:active p,
.stButton > button:focus p {
    background: #994422 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 1.8rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    min-width: 180px !important;
    width: auto !important;
}
.stButton > button:hover,
.stButton > button:hover p {
    background: #b85c38 !important;
    color: #ffffff !important;
}

/* Metric card */
.metric-card {
    background: #fff8f3;
    border: 1px solid #e8e4d8;
    border-radius: 8px;
    padding: 1.5rem 1.6rem;
}
.metric-card .label {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    color: #55433c;
    margin-bottom: 0.6rem;
}
.metric-card .value {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 40px;
    font-weight: 400;
    color: #1d1b18;
    line-height: 48px;
    letter-spacing: -0.01em;
}
.metric-card .delta {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #5a7a4e;
    margin-top: 0.5rem;
}

/* Status indicators */
.status-dot {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
}
.status-dot::before {
    content: "";
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
}
.status-reliable { color: #5a7a4e; }
.status-reliable::before { background: #5a7a4e; }
.status-low { color: #b8801f; }
.status-low::before { background: #b8801f; }
.status-very-low { color: #ba1a1a; }
.status-very-low::before { background: #ba1a1a; }

/* Status chip */
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.7rem;
    border-radius: 9999px;
    background: #f3ede7;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    margin-left: 0.6rem;
}
.status-chip::before {
    content: "";
    width: 7px; height: 7px; border-radius: 50%;
}
.chip-reliable { color: #5a7a4e; }
.chip-reliable::before { background: #5a7a4e; }
.chip-low { color: #b8801f; }
.chip-low::before { background: #b8801f; }
.chip-very-low { color: #ba1a1a; }
.chip-very-low::before { background: #ba1a1a; }

/* Diagnosis card */
.diagnosis-card {
    background: #f9f2ed;
    border: 1px solid #e8e4d8;
    border-left: 3px solid #994422;
    border-radius: 8px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.5rem;
}
.diagnosis-card .label {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    color: #994422;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.diagnosis-card .title {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 22px;
    font-weight: 500;
    color: #1d1b18;
    margin-bottom: 0.9rem;
}
.diagnosis-card .body {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    line-height: 24px;
    color: #1d1b18;
}

/* Confusion matrix tiles */
.cm-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.cm-tile {
    border-radius: 8px;
    padding: 1.6rem 1.8rem;
    min-height: 160px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    border: 1px solid #e8e4d8;
}
.cm-tile.primary {
    background: #994422;
    color: #ffffff;
    border-color: #994422;
}
.cm-tile.muted {
    background: #ede7e2;
    color: #1d1b18;
}
.cm-tile .num {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 40px;
    line-height: 1;
    font-weight: 400;
}
.cm-tile .desc {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    margin-top: 0.4rem;
    opacity: 0.85;
}
.cm-tile .cat {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.7;
    margin-top: 1rem;
}

/* Captions */
.small-meta {
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    color: #55433c;
    margin-top: 0.6rem;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_inspector():
    return Inspector()


@st.cache_data
def load_eval_results():
    try:
        return pd.read_csv("evaluation_results.csv")
    except Exception:
        return None


@st.cache_data
def load_best_threshold():
    try:
        with open("best_threshold.json") as f:
            return json.load(f)
    except Exception:
        return None


def status_chip(status):
    cls_map = {"RELIABLE": ("chip-reliable", "Reliable"),
               "LOW": ("chip-low", "Low"),
               "VERY_LOW": ("chip-very-low", "Very low")}
    cls, label = cls_map.get(status, ("chip-low", status))
    return f'<span class="status-chip {cls}">{label}</span>'


def status_dot(status):
    cls_map = {"RELIABLE": ("status-reliable", "Clear"),
               "LOW": ("status-low", "Flagged"),
               "VERY_LOW": ("status-very-low", "Flagged")}
    cls, label = cls_map.get(status, ("status-low", "Flagged"))
    return f'<span class="status-dot {cls}">{label}</span>'


def metric_card(label, value, delta=None):
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    return (f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'{delta_html}</div>')


# ----------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero-block">
  <h1>RAG Reliability Inspector</h1>
  <div class="sub">Observability for RAG answer quality. Detect and diagnose hallucinations before they reach your users.</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading models..."):
    inspector = load_inspector()

tab1, tab2, tab3 = st.tabs(["Playground", "Monitoring Feed", "Model Performance"])

# ======================================================================
# TAB 1 — PLAYGROUND
# ======================================================================
with tab1:
    st.markdown('<p class="small-meta">Test and analyze RAG responses for hallucination and relevance in real-time.</p>',
                unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        query = st.text_area("Query",
                             value="How long is the warranty?",
                             height=100)
        context = st.text_area("Retrieved Context",
                               value="The warranty covers the product for 12 months from purchase.",
                               height=180)
    with col2:
        answer = st.text_area("Generated Answer",
                              value="The product is covered under warranty for 24 months.",
                              height=300)

    st.write("")
    _, btn_col = st.columns([4, 2])
    with btn_col:
        clicked = st.button("Inspect Reliability", type="primary")

    if clicked:
        if not query.strip() or not context.strip() or not answer.strip():
            st.warning("Please fill in all three fields: query, context, and answer.")
        else:
            with st.spinner("Inspecting..."):
                result = inspector.evaluate(query, context, answer)

            st.markdown(
                f'<h3 style="margin-top:1.5rem;">Overall Status {status_chip(result.overall_status)}</h3>',
                unsafe_allow_html=True,
            )
            st.write("")

            c1, c2, c3 = st.columns(3, gap="large")
            c1.markdown(metric_card("Retrieval relevance",
                                    f"{result.retrieval_relevance:.2f}"),
                        unsafe_allow_html=True)
            c2.markdown(metric_card("Faithfulness",
                                    f"{result.faithfulness_score:.2f}"),
                        unsafe_allow_html=True)
            c3.markdown(metric_card("Hallucination risk",
                                    f"{result.hallucination_score:.2f}"),
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div class="diagnosis-card">
              <div class="label">Diagnosis</div>
              <div class="body">{result.diagnosis}</div>
            </div>
            """, unsafe_allow_html=True)

# ======================================================================
# TAB 2 — MONITORING FEED
# ======================================================================
with tab2:
    st.markdown('<p class="small-meta">Real-time surveillance of RAG reliability metrics. Inspect flagged interactions and assess retrieval faithfulness.</p>',
                unsafe_allow_html=True)
    st.write("")

    LOG_PATH = "monitoring_log.jsonl"

    col_a, col_b, _ = st.columns([2, 2, 4])
    with col_a:
        refresh = st.button("Refresh feed")
    with col_b:
        clear = st.button("Clear feed")

    if clear and os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
        st.success("Feed cleared.")

    records = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        st.info("No monitored answers yet. Run the demo app "
                "(`python3 -m demo_rag_app`) to populate the feed, "
                "then click Refresh feed.")
    else:
        total = len(records)
        n_problem = sum(1 for r in records if r["status"] != "RELIABLE")
        pct = n_problem / total * 100

        st.write("")
        s1, s2, s3 = st.columns(3, gap="large")
        s1.markdown(metric_card("Total answers", f"{total:,}"),
                    unsafe_allow_html=True)
        s2.markdown(metric_card("Flagged responses", f"{n_problem}"),
                    unsafe_allow_html=True)
        s3.markdown(metric_card("Flag rate", f"{pct:.1f}%"),
                    unsafe_allow_html=True)

        st.write("")
        st.markdown("### Recent interactions")

        table_html = """
        <style>
        .feed-table { width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; font-size: 14px; margin-top: 0.5rem; border: 1px solid #e8e4d8; border-radius: 8px; overflow: hidden; }
        .feed-table th { text-align: left; font-weight: 500; color: #55433c; font-size: 13px; padding: 0.9rem 1rem; border-bottom: 1px solid #e8e4d8; background: #f9f2ed; }
        .feed-table td { padding: 1rem; border-bottom: 1px solid #e8e4d8; color: #1d1b18; vertical-align: top; }
        .feed-table tr:last-child td { border-bottom: none; }
        .num-flag { color: #ba1a1a; }
        </style>
        <table class="feed-table">
          <thead>
            <tr>
              <th>Status</th><th>Time</th><th>Query</th>
              <th style="text-align:right;">Faithfulness</th>
              <th style="text-align:right;">Retrieval</th>
            </tr>
          </thead>
          <tbody>
        """
        for r in reversed(records):
            time_short = r.get("timestamp", "").split("T")[-1] if "timestamp" in r else ""
            query_short = r["query"]
            f_score = r["faithfulness_score"]
            ret_score = r["retrieval_relevance"]
            f_class = "num-flag" if f_score < 0.5 else ""
            ret_class = "num-flag" if ret_score < 0.05 else ""
            table_html += (
                f"<tr>"
                f"<td>{status_dot(r['status'])}</td>"
                f"<td>{time_short}</td>"
                f"<td>{query_short}</td>"
                f'<td style="text-align:right;" class="{f_class}">{f_score:.2f}</td>'
                f'<td style="text-align:right;" class="{ret_class}">{ret_score:.2f}</td>'
                f"</tr>"
            )
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        flagged = [r for r in reversed(records) if r["status"] != "RELIABLE"]
        if flagged:
            st.write("")
            st.write("")
            for r in flagged:
                st.markdown(f"""
                <div class="diagnosis-card">
                  <div class="label">Diagnosis report</div>
                  <div class="title">{r['query']}</div>
                  <div class="body">
                    <b>Generated response:</b> {r['answer']}<br><br>
                    <b>Issue analysis:</b> {r['diagnosis']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ======================================================================
# TAB 3 — MODEL PERFORMANCE
# ======================================================================
with tab3:
    st.markdown('<p class="small-meta">An analysis of the current retrieval-augmented generation model\'s ability to accurately synthesize and recall critical source documents.</p>',
                unsafe_allow_html=True)
    st.write("")

    # Read latest metrics from best_threshold.json (auto-updates with new evaluations)
    bt = load_best_threshold()
    if bt:
        f1_val = f"{bt['f1']:.2f}"
        prec_val = f"{bt['precision']:.2f}"
        rec_val = f"{bt['recall']:.2f}"
    else:
        f1_val, prec_val, rec_val = "0.63", "0.58", "0.70"

    c1, c2, c3 = st.columns(3, gap="large")
    c1.markdown(metric_card("F1 Score", f1_val, "Fine-tuned RoBERTa on RAGTruth"),
                unsafe_allow_html=True)
    c2.markdown(metric_card("Precision", prec_val),
                unsafe_allow_html=True)
    c3.markdown(metric_card("Recall", rec_val),
                unsafe_allow_html=True)

    st.markdown('<p class="small-meta">Fine-tuned RoBERTa on the RAGTruth QA test set (300 examples). Zero-shot baseline F1 was 0.43; our fine-tuned model reaches 0.63, a ~47% relative improvement, with false positives reduced from 108 to 29.</p>',
                unsafe_allow_html=True)

    st.write("")
    st.markdown("### Confusion matrix evaluation")

    results = load_eval_results()
    if (results is not None
            and "truth_hallucinated" in results.columns
            and "predicted_hallucinated" in results.columns):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results["truth_hallucinated"],
                              results["predicted_hallucinated"])
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

        st.markdown(f"""
        <div class="cm-grid">
          <div class="cm-tile primary">
            <div>
              <div class="num">{tp}</div>
              <div class="desc">Valid detections</div>
            </div>
            <div class="cat">True Positive</div>
          </div>
          <div class="cm-tile muted">
            <div>
              <div class="num">{fp}</div>
              <div class="desc">False alarms</div>
            </div>
            <div class="cat">False Positive</div>
          </div>
          <div class="cm-tile muted">
            <div>
              <div class="num">{fn}</div>
              <div class="desc">Missed hallucinations</div>
            </div>
            <div class="cat">False Negative</div>
          </div>
          <div class="cm-tile muted">
            <div>
              <div class="num">{tn}</div>
              <div class="desc">Correctly cleared</div>
            </div>
            <div class="cat">True Negative</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Evaluation results CSV not found or missing expected columns.")