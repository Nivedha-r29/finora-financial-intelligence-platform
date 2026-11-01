"""Streamlit entry point for the Finora research dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.data_loader import PROJECT_ROOT, load_phase8_artifacts
from dashboard.pages import (
    render_architecture,
    render_experiment_results,
    render_federated_learning,
    render_fraud_detection,
    render_iid_analysis,
    render_model_comparison,
    render_overview,
    render_privacy_security,
)


@st.cache_data(show_spinner=False)
def _load_artifacts(project_root: str) -> dict:
    return load_phase8_artifacts(Path(project_root))


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp {
          background:
            radial-gradient(ellipse at 82% 0%, rgba(32, 113, 138, .16), transparent 34%),
            linear-gradient(145deg, #07111f 0%, #0a1626 55%, #0c1a2c 100%);
          color: #eef5ff;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #0b1728 0%, #101e31 100%);
          border-right: 1px solid rgba(133, 171, 203, .15);
        }
        [data-testid="stSidebar"] * { color: #dce8f7; }
        h1, h2, h3 { color: #f3f8ff; letter-spacing: -.025em; }
        p, label, .stCaption { color: #b7c7da; }
        .finora-brand {
          font-size: 2rem; font-weight: 850; letter-spacing: .18em;
          color: #32d6c5; margin: .2rem 0 0;
        }
        .finora-subtitle {
          font-size: .78rem; line-height: 1.45; color: #a9bbd0;
          margin: 0 0 1.2rem;
        }
        .metric-card, .notice-card, .flow-card {
          background: linear-gradient(145deg, rgba(20, 39, 60, .96), rgba(13, 28, 46, .96));
          border: 1px solid rgba(93, 134, 168, .23);
          border-radius: 16px; padding: 1rem 1.05rem; margin: .35rem 0 .8rem;
          box-shadow: 0 10px 24px rgba(0, 0, 0, .14);
        }
        .metric-label { font-size: .78rem; color: #9fb3c9; min-height: 1.3rem; }
        .metric-value { font-size: 1.35rem; font-weight: 750; color: #f4f8ff; margin: .25rem 0; }
        .metric-detail { font-size: .72rem; color: #8299b2; }
        .flow-row {
          display: flex; flex-wrap: wrap; align-items: center;
          justify-content: center; gap: .45rem; margin: 1rem 0;
        }
        .flow-card { min-width: 150px; text-align: center; color: #f2f7ff; margin: 0; }
        .flow-card span { font-size: .72rem; color: #9fb3c9; }
        .flow-arrow { color: #32d6c5; font-size: 1.4rem; font-weight: 800; }
        .secure-flow {
          display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: .45rem; margin: .7rem 0 1rem;
        }
        .secure-flow div, .secure-flow strong {
          border: 1px solid rgba(93, 134, 168, .23); border-radius: 10px;
          padding: .65rem; background: rgba(20, 39, 60, .75); color: #dce8f7;
        }
        .secure-flow strong { color: #32d6c5; text-align: center; }
        .notice-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .65rem; }
        .notice-card { color: #b7c7da; font-size: .9rem; }
        .notice-card b { color: #32d6c5; }
        div[data-testid="stMetric"] {
          background: rgba(19, 38, 58, .78); padding: .9rem;
          border-radius: 14px; border: 1px solid rgba(93, 134, 168, .2);
        }
        div[data-testid="stDataFrame"] { border: 1px solid rgba(93, 134, 168, .18); border-radius: 12px; }
        .stButton > button, .stDownloadButton > button {
          border-radius: 10px; border: 1px solid rgba(50, 214, 197, .42);
          background: rgba(50, 214, 197, .11); color: #e9fffc; font-weight: 650;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
          border-color: #32d6c5; color: #fff; background: rgba(50, 214, 197, .2);
        }
        @media (max-width: 800px) {
          .notice-grid { grid-template-columns: 1fr; }
          .flow-card { min-width: 120px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


PAGES = {
    "Executive overview": render_overview,
    "Fraud detection demo": render_fraud_detection,
    "Federated learning": render_federated_learning,
    "Privacy & security": render_privacy_security,
    "Model comparison": render_model_comparison,
    "IID vs Non-IID": render_iid_analysis,
    "System architecture": render_architecture,
    "Experiment results": render_experiment_results,
}


def main() -> None:
    st.set_page_config(
        page_title="FINORA · Federated Fraud Analytics",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()
    with st.sidebar:
        st.markdown('<div class="finora-brand">FINORA</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="finora-subtitle">Privacy-Preserving Federated Learning<br>'
            "for Financial Fraud Detection</div>",
            unsafe_allow_html=True,
        )
        selected_page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption("Historical results · Phase 8")
        st.caption("Research demonstration · Not a banking system")

    st.markdown(
        f'<div class="finora-subtitle">FINORA / {selected_page.upper()}</div>',
        unsafe_allow_html=True,
    )
    artifacts = _load_artifacts(str(PROJECT_ROOT))
    PAGES[selected_page](artifacts)
    st.divider()
    st.caption(
        "Finora is a research visualization. Saved experiments are historical; "
        "prediction demos run locally from saved model checkpoints."
    )


if __name__ == "__main__":
    main()
