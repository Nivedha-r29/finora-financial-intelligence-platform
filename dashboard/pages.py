"""Page renderers for the Finora historical-results dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from dashboard.data_loader import (
    DashboardDataError,
    load_demo_transaction,
    load_model_checkpoint,
    load_predictions,
    load_preprocessing,
    predict_transaction,
)


METHOD_LABELS = {
    "centralized": "Centralized",
    "fedavg": "FedAvg",
    "dp_fedavg": "DP-FedAvg",
    "secure_fedavg": "Secure FedAvg",
    "secure_dp_fedavg": "Secure DP-FedAvg",
}
PREFERRED_METHOD_ORDER = list(METHOD_LABELS)
COLORS = {
    "centralized": "#8fa3b8",
    "fedavg": "#32d6c5",
    "dp_fedavg": "#f0b857",
    "secure_fedavg": "#5c9dff",
    "secure_dp_fedavg": "#c890ff",
}


def _show_artifact_errors(data: dict[str, Any], keys: list[str]) -> None:
    for key in keys:
        message = data.get("errors", {}).get(key)
        if message:
            st.warning(message)


def _require_frame(data: dict[str, Any], key: str, label: str) -> pd.DataFrame | None:
    frame = data.get(key)
    if not isinstance(frame, pd.DataFrame):
        _show_artifact_errors(data, [key])
        st.info(f"{label} is unavailable in the saved experiment artifacts.")
        return None
    return frame


def _title(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def _successful_metrics(data: dict[str, Any]) -> pd.DataFrame | None:
    frame = _require_frame(data, "final_metrics", "Final metrics")
    if frame is None:
        return None
    return frame.loc[frame["status"].eq("success")].copy()


def _filter_partition(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
    selected = frame.loc[frame["partition"].isin([partition, "not_applicable"])].copy()
    selected["_method_order"] = selected["method"].map(
        {method: index for index, method in enumerate(PREFERRED_METHOD_ORDER)}
    )
    return selected.sort_values("_method_order").drop(columns="_method_order")


def _metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def render_overview(data: dict[str, Any]) -> None:
    st.title("Executive overview")
    st.caption(
        "A research demonstration of collaborative fraud-model training. "
        "Historical Phase 8 results are read from saved artifacts; nothing is retrained here."
    )
    statistics = data.get("dataset_statistics")
    metrics = _successful_metrics(data)
    if isinstance(statistics, dict):
        frauds = statistics.get("fraud_count")
        legitimate = statistics.get("legitimate_count")
        rows = statistics.get("rows")
        clients = int((data.get("experiment_config") or {}).get("num_clients", 0))
        rounds = int((data.get("experiment_config") or {}).get("rounds", 0))
        modes = (
            metrics.loc[metrics["method"].ne("centralized"), "method"].nunique()
            if metrics is not None
            else 0
        )
        best = metrics.loc[metrics["pr_auc_average_precision"].idxmax()] if metrics is not None and not metrics.empty else None
        best_label = f"{_title(best['method'])} · {best['pr_auc_average_precision']:.4f}" if best is not None else "Unavailable"
        card_columns = st.columns(4)
        cards = [
            ("Transactions", f"{rows:,}" if isinstance(rows, (int, float)) else "Unavailable", "Kaggle source dataset"),
            ("Fraud cases", f"{frauds:,}" if isinstance(frauds, (int, float)) else "Unavailable", "Class = 1"),
            ("Legitimate", f"{legitimate:,}" if isinstance(legitimate, (int, float)) else "Unavailable", "Class = 0"),
            ("Institutions", str(clients) if clients else "Unavailable", "Simulated clients"),
            ("Best observed PR-AUC", best_label, "Average precision; saved results"),
            ("FedAvg rounds", str(rounds) if rounds else "Unavailable", "Historical experiment setting"),
            ("Federated/privacy modes", str(modes) if modes else "Unavailable", "FedAvg, DP, secure, secure + DP"),
            ("Threshold", str((data.get("experiment_config") or {}).get("decision_threshold", "Unavailable")), "Fixed; not test-tuned"),
        ]
        for index, card in enumerate(cards):
            with card_columns[index % 4]:
                _metric_card(*card)

        class_distribution = statistics.get("class_distribution", {})
        if class_distribution:
            distribution = pd.DataFrame(
                [
                    {"Class": "Legitimate", "Transactions": int(class_distribution.get("0", 0))},
                    {"Class": "Fraud", "Transactions": int(class_distribution.get("1", 0))},
                ]
            )
            st.subheader("Dataset class distribution")
            figure = px.pie(
                distribution,
                names="Class",
                values="Transactions",
                hole=0.68,
                color="Class",
                color_discrete_map={"Legitimate": "#32d6c5", "Fraud": "#ff7187"},
            )
            figure.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend_title_text="",
            )
            left, right = st.columns([1, 1.4])
            with left:
                st.plotly_chart(figure, width="stretch")
            with right:
                st.subheader("Observed model performance")
                if metrics is not None:
                    summary = metrics.loc[
                        metrics["partition"].eq("iid") | metrics["method"].eq("centralized"),
                        ["method", "accuracy", "precision", "recall", "f1_score", "pr_auc_average_precision"],
                    ].copy()
                    summary["method"] = summary["method"].map(_title)
                    summary = summary.rename(
                        columns={"pr_auc_average_precision": "PR-AUC"}
                    )
                    st.dataframe(summary, hide_index=True, width="stretch")
                    st.caption("IID federated results plus the partition-independent centralized baseline.")
                _show_artifact_errors(data, ["dataset_statistics", "final_metrics"])
    else:
        _show_artifact_errors(data, ["dataset_statistics"])

    st.subheader("How Finora works")
    st.markdown(
        """
        <div class="flow-row">
          <div class="flow-card">Five simulated institutions<br><span>Raw transactions stay local</span></div>
          <div class="flow-arrow">→</div>
          <div class="flow-card">Local training<br><span>Each client updates a model</span></div>
          <div class="flow-arrow">→</div>
          <div class="flow-card">Privacy options<br><span>DP-SGD and/or masked updates</span></div>
          <div class="flow-arrow">→</div>
          <div class="flow-card">Federated server<br><span>Aggregates model updates</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Federated learning is not, by itself, a guarantee that model updates reveal no information. "
        "Differential privacy and secure aggregation address different risks; this is an in-memory simulation."
    )


@st.cache_resource(show_spinner=False)
def _cached_model(checkpoint_path: str, feature_count: int):
    return load_model_checkpoint(Path(checkpoint_path), feature_count)


@st.cache_data(show_spinner=False)
def _cached_preprocessing(project_root: str):
    return load_preprocessing(Path(project_root))


@st.cache_data(show_spinner="Selecting an anonymized held-out sample...")
def _cached_demo_transaction(project_root: str):
    return load_demo_transaction(Path(project_root))


def render_fraud_detection(data: dict[str, Any]) -> None:
    st.title("Fraud detection demonstration")
    st.warning(
        "Research demonstration only — not a real banking decision system. "
        "Predictions are model outputs, not financial advice or a production fraud decision."
    )
    metrics = _successful_metrics(data)
    if metrics is None:
        return
    root = Path(data["root"])
    models_dir = data["phase8_root"] / "models"
    choices = []
    for row in metrics.itertuples(index=False):
        checkpoint_name = "centralized.pt" if row.method == "centralized" else f"{row.experiment_id}.pt"
        path = models_dir / checkpoint_name
        if path.is_file():
            choices.append((f"{_title(row.method)} · {row.partition}", path))
    if not choices:
        st.error(f"No valid Phase 8 model checkpoint is available in {models_dir}.")
        return

    choice_labels = [label for label, _ in choices]
    chosen_label = st.selectbox("Saved model checkpoint", choice_labels)
    checkpoint_path = choices[choice_labels.index(chosen_label)][1]
    try:
        preprocessing = _cached_preprocessing(str(root))
        model, _ = _cached_model(str(checkpoint_path), len(preprocessing["feature_columns"]))
    except DashboardDataError as exc:
        st.error(str(exc))
        return

    mode = st.radio(
        "Transaction input",
        ("Manual entry", "Anonymized held-out sample"),
        horizontal=True,
    )
    submitted_values: dict[str, float] | None = None
    if mode == "Manual entry":
        st.caption("Enter the raw, pre-scaling values for the recorded model features.")
        columns = preprocessing["feature_columns"]
        with st.form("manual_transaction"):
            field_columns = st.columns(3)
            values = {}
            for index, feature in enumerate(columns):
                kwargs: dict[str, Any] = {
                    "label": feature,
                    "value": 0.0,
                    "format": "%.6f",
                    "key": f"manual_{feature}",
                }
                if feature in {"Time", "Amount"}:
                    kwargs["min_value"] = 0.0
                with field_columns[index % 3]:
                    values[feature] = st.number_input(**kwargs)
            submitted = st.form_submit_button("Evaluate transaction", type="primary")
        if submitted:
            submitted_values = values
    else:
        st.caption(
            "A reproducibly selected record from the held-out split is read on demand. "
            "Its source features and identity are not displayed."
        )
        if st.button("Predict on real held-out sample", type="primary"):
            try:
                submitted_values = _cached_demo_transaction(str(root))
            except DashboardDataError as exc:
                st.error(str(exc))

    if submitted_values is not None:
        try:
            probability = predict_transaction(submitted_values, model, preprocessing)
        except DashboardDataError as exc:
            st.error(str(exc))
            return
        threshold = preprocessing["threshold"]
        is_fraud = probability >= threshold
        left, right = st.columns(2)
        with left:
            st.metric("Fraud probability", f"{probability:.2%}")
        with right:
            st.metric("Model classification", "Potential fraud" if is_fraud else "Likely legitimate")
        st.progress(probability)
        st.caption(
            f"Threshold: {threshold:.2f} · Checkpoint: {checkpoint_path.name}. "
            "Probability is produced by this saved checkpoint after applying the recorded feature order "
            "and training-fitted scaler."
        )
        st.info(
            "Historical evaluation used a fixed threshold of 0.5. A single demonstration score "
            "does not establish real-world model performance."
        )


def render_federated_learning(data: dict[str, Any]) -> None:
    st.title("Federated learning")
    st.caption(
        "Historical Phase 8 run summaries. These are not live training sessions."
    )
    distribution = st.selectbox("Client distribution", ("iid", "non_iid"))
    clients = data.get("client_distribution")
    if not isinstance(clients, pd.DataFrame):
        _show_artifact_errors(data, ["client_distribution"])
        return
    client_rows = clients.loc[clients["partition"].eq(distribution)].copy()
    if client_rows.empty:
        st.info(f"No saved client statistics are available for {distribution}.")
        return
    client_rows["fraud_percentage"] = client_rows["fraud_rate"] * 100
    successful = _successful_metrics(data)
    partition_runs = (
        successful.loc[successful["partition"].eq(distribution)]
        if successful is not None
        else pd.DataFrame()
    )
    all_completed = not partition_runs.empty and partition_runs["status"].eq("success").all()
    client_rows["historical_status"] = (
        "Included in successful saved runs" if all_completed else "Status not fully available"
    )
    client_rows["individual_local_metrics"] = "Not recorded in Phase 8 outputs"
    display = client_rows[
        [
            "client_id",
            "sample_count",
            "fraud_count",
            "fraud_percentage",
            "historical_status",
            "individual_local_metrics",
        ]
    ].rename(
        columns={
            "client_id": "Client",
            "sample_count": "Transactions",
            "fraud_count": "Fraud cases",
            "fraud_percentage": "Fraud %",
            "historical_status": "Historical run status",
            "individual_local_metrics": "Client-level metrics",
        }
    )
    st.subheader(f"Five simulated institutions · {distribution.upper()}")
    st.dataframe(display, hide_index=True, width="stretch")
    figure = px.bar(
        client_rows,
        x="client_id",
        y="sample_count",
        color="fraud_percentage",
        color_continuous_scale=["#32d6c5", "#ff7187"],
        hover_data={"fraud_count": True, "fraud_percentage": ":.3f"},
        labels={"client_id": "Institution", "sample_count": "Training transactions", "fraud_percentage": "Fraud %"},
    )
    figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, width="stretch")

    st.subheader("Recorded training workflow")
    st.markdown(
        "1. Server distributes the global model → 2. each client trains locally → "
        "3. model updates are submitted → 4. FedAvg or secure aggregation produces a global update → "
        "5. the global model is evaluated."
    )
    rounds = _require_frame(data, "per_round_metrics", "Per-round metrics")
    if rounds is not None:
        selected = rounds.loc[rounds["partition"].eq(distribution)].copy()
        if not selected.empty:
            selected["method_label"] = selected["method"].map(_title)
            left, right = st.columns(2)
            with left:
                chart = px.line(
                    selected,
                    x="round",
                    y="accuracy",
                    color="method_label",
                    markers=True,
                    title="Global accuracy across communication rounds",
                    labels={"round": "Communication round", "accuracy": "Accuracy", "method_label": "Method"},
                )
                chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(chart, width="stretch")
            with right:
                chart = px.line(
                    selected,
                    x="round",
                    y="test_loss",
                    color="method_label",
                    markers=True,
                    title="Global test loss across communication rounds",
                    labels={"round": "Communication round", "test_loss": "Test loss", "method_label": "Method"},
                )
                chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(chart, width="stretch")
    st.info(
        "Per-client loss and accuracy scalars were not saved in the Phase 8 artifact set; "
        "they are shown as unavailable rather than reconstructed."
    )


def render_privacy_security(data: dict[str, Any]) -> None:
    st.title("Privacy & security")
    st.caption("Differential privacy and secure aggregation protect different parts of the workflow.")
    config = data.get("experiment_config") or {}
    left, right = st.columns(2)
    with left:
        st.subheader("Differential Privacy · DP-SGD")
        st.markdown(
            "For each local example, DP-SGD computes gradients, clips their norm, "
            "adds calibrated Gaussian noise to the batch gradient, and applies the optimizer update."
        )
        st.metric("Maximum gradient norm", config.get("dp_max_grad_norm", "Unavailable"))
        st.metric("Noise multiplier", config.get("dp_noise_multiplier", "Unavailable"))
        st.metric("Delta", config.get("dp_delta", "Unavailable"))
        st.error(
            "Epsilon values below are approximate estimates from the existing lightweight accountant. "
            "They are not a formally certified privacy guarantee."
        )
        partition = st.selectbox(
            "Privacy-accounting partition",
            ("iid", "non_iid"),
            key="privacy_partition",
        )
        privacy = data.get("privacy_accounting")
        if isinstance(privacy, dict):
            rows = []
            for method in ("dp_fedavg", "secure_dp_fedavg"):
                key = f"{method}_{partition}"
                record = privacy.get(key)
                if not isinstance(record, dict):
                    continue
                for client_id, client in record.get("per_client", {}).items():
                    rows.append(
                        {
                            "Method": _title(method),
                            "Client": client_id,
                            "Approx. epsilon": client.get("epsilon_estimate"),
                            "Delta": client.get("delta"),
                            "Steps": client.get("steps"),
                        }
                    )
            if rows:
                frame = pd.DataFrame(rows)
                chart = px.bar(
                    frame,
                    x="Client",
                    y="Approx. epsilon",
                    color="Method",
                    barmode="group",
                    title="Approximate per-client epsilon estimates",
                    color_discrete_sequence=["#f0b857", "#c890ff"],
                )
                chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(chart, width="stretch")
                st.dataframe(frame, hide_index=True, width="stretch")
        else:
            _show_artifact_errors(data, ["privacy_accounting"])
            st.info("Privacy-accounting results are unavailable.")
        st.caption(
            "Accountant assumptions—especially sampling under shuffled mini-batches—have not been formally validated."
        )
    with right:
        st.subheader("Secure aggregation · pairwise masks")
        st.markdown(
            """
            - Clients establish pairwise secrets with X25519 and derive mask streams through HKDF-SHA-256.
            - Each update is fixed-point quantized and masked with signed pairwise masks.
            - Opposite masks cancel when the masked values are summed modulo the configured modulus.
            - Shamir shares support recovery from permitted pre-upload client dropouts.
            """
        )
        st.markdown(
            """
            <div class="secure-flow">
              <div>Client 1 → masked update</div>
              <div>Client 2 → masked update</div>
              <div>Client 3 → masked update</div>
              <div>Client 4 → masked update</div>
              <div>Client 5 → masked update</div>
              <strong>↓ aggregate masked values</strong>
              <div>Server obtains the aggregate update</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.success(
            "In the secure aggregation simulation, the server aggregates masked submissions rather than "
            "directly receiving each individual plaintext update."
        )
        st.warning(
            "The server still observes participation, sample counts, timing, and monitoring scalars. "
            "The implementation is in-memory; authenticated public keys and secure network transport "
            "are not production-ready."
        )
        st.info("Secure aggregation does not provide differential privacy.")


def _curve_plot(
    predictions: pd.DataFrame,
    experiment_ids: list[str],
    curve_kind: str,
) -> go.Figure | None:
    figure = go.Figure()
    for experiment_id in experiment_ids:
        sample = predictions.loc[predictions["experiment_id"].eq(experiment_id)]
        if sample.empty or sample["label"].nunique() != 2:
            continue
        labels = sample["label"].astype(int).to_numpy()
        probabilities = sample["fraud_probability"].astype(float).to_numpy()
        if curve_kind == "roc":
            x_values, y_values, _ = roc_curve(labels, probabilities)
            name = experiment_id.replace("_", " ").title()
            figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", name=name))
        else:
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            name = experiment_id.replace("_", " ").title()
            figure.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
    if not figure.data:
        return None
    if curve_kind == "roc":
        figure.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash", "color": "#65748b"})
        )
        figure.update_layout(title="ROC curves from saved test predictions", xaxis_title="False positive rate", yaxis_title="True positive rate")
    else:
        figure.update_layout(title="Precision–Recall curves from saved test predictions", xaxis_title="Recall", yaxis_title="Precision")
    figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    return figure


def render_model_comparison(data: dict[str, Any]) -> None:
    st.title("Model comparison")
    st.caption("Only recorded Phase 8 experiment outputs are plotted; no missing results are synthesized.")
    metrics = _successful_metrics(data)
    if metrics is None:
        return
    partition = st.selectbox("Federated distribution", ("iid", "non_iid"), key="comparison_partition")
    filtered = _filter_partition(metrics, partition)
    if filtered.empty:
        st.info(f"No successful metrics are available for {partition}.")
        return
    available_methods = filtered["method"].drop_duplicates().tolist()
    chosen_methods = st.multiselect(
        "Methods to compare",
        available_methods,
        default=available_methods,
        format_func=_title,
    )
    filtered = filtered.loc[filtered["method"].isin(chosen_methods)].copy()
    if filtered.empty:
        st.info("Select at least one method.")
        return
    display = filtered[
        [
            "method",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "pr_auc_average_precision",
            "false_positive_rate",
            "false_negative_rate",
            "training_duration_seconds",
        ]
    ].copy()
    display["method"] = display["method"].map(_title)
    display = display.rename(columns={"pr_auc_average_precision": "pr_auc (average precision)"})
    st.dataframe(display, hide_index=True, width="stretch")

    chart_data = filtered.melt(
        id_vars=["method"],
        value_vars=["precision", "recall", "f1_score", "pr_auc_average_precision"],
        var_name="Metric",
        value_name="Score",
    )
    chart_data["Method"] = chart_data["method"].map(_title)
    chart_data["Metric"] = chart_data["Metric"].replace(
        {"f1_score": "F1", "pr_auc_average_precision": "PR-AUC"}
    )
    chart = px.bar(
        chart_data,
        x="Metric",
        y="Score",
        color="Method",
        barmode="group",
        title="Fraud-sensitive metrics at the fixed 0.5 threshold",
        color_discrete_map={_title(method): color for method, color in COLORS.items()},
    )
    chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1])
    st.plotly_chart(chart, width="stretch")
    st.caption("PR-AUC is average precision and is threshold-independent; the displayed precision/recall/F1 use threshold 0.5.")

    try:
        predictions = load_predictions(Path(data["root"]))
    except DashboardDataError as exc:
        st.warning(str(exc))
        predictions = None
    if predictions is not None:
        prediction_ids = filtered["experiment_id"].tolist()
        left, right = st.columns(2)
        with left:
            roc_figure = _curve_plot(predictions, prediction_ids, "roc")
            if roc_figure is not None:
                st.plotly_chart(roc_figure, width="stretch")
            else:
                st.info("ROC curves are unavailable for the selected saved predictions.")
        with right:
            pr_figure = _curve_plot(predictions, prediction_ids, "pr")
            if pr_figure is not None:
                st.plotly_chart(pr_figure, width="stretch")
            else:
                st.info("Precision–Recall curves are unavailable for the selected saved predictions.")

    left, right = st.columns(2)
    with left:
        figure = px.bar(
            filtered,
            x="experiment_id",
            y="training_duration_seconds",
            color="method",
            title="Recorded training duration",
            labels={"experiment_id": "Experiment", "training_duration_seconds": "Seconds"},
            color_discrete_map=COLORS,
        )
        figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch")
    with right:
        communication = _require_frame(data, "communication_overhead", "Communication estimates")
        if communication is not None:
            selected_communication = communication.loc[
                communication["experiment_id"].isin(filtered["experiment_id"])
            ]
            figure = px.bar(
                selected_communication,
                x="experiment_id",
                y="estimated_total_communication_bytes",
                title="Estimated communication volume",
                labels={"experiment_id": "Experiment", "estimated_total_communication_bytes": "Estimated bytes"},
                color="experiment_id",
            )
            figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure, width="stretch")

    st.subheader("Confusion matrices")
    matrix_columns = st.columns(min(3, len(filtered)))
    for index, row in enumerate(filtered.itertuples(index=False)):
        try:
            matrix = np.asarray(json.loads(row.confusion_matrix), dtype=int)
            if matrix.shape != (2, 2):
                raise ValueError("matrix must be 2x2")
        except (json.JSONDecodeError, TypeError, ValueError):
            with matrix_columns[index % len(matrix_columns)]:
                st.warning(f"Confusion matrix unavailable for {row.experiment_id}.")
            continue
        figure = px.imshow(
            matrix,
            text_auto=True,
            x=["Predicted legitimate", "Predicted fraud"],
            y=["Actual legitimate", "Actual fraud"],
            color_continuous_scale=["#142033", "#32d6c5"],
            title=f"{_title(row.method)} · {row.partition}",
            aspect="auto",
        )
        figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        with matrix_columns[index % len(matrix_columns)]:
            st.plotly_chart(figure, width="stretch")

    st.info(
        "The DP methods' observed threshold metrics can be strongly affected by clipping, noise, and the "
        "training-derived class weight. Secure aggregation changes update visibility, not model privacy by itself."
    )


def render_iid_analysis(data: dict[str, Any]) -> None:
    st.title("IID vs Non-IID analysis")
    st.markdown(
        "Financial institutions may have different customer populations, channels, and fraud patterns. "
        "This experiment uses a constrained per-class Dirichlet allocation for its non-IID partition."
    )
    clients = _require_frame(data, "client_distribution", "Client distribution")
    metrics = _successful_metrics(data)
    if clients is None:
        return
    clients = clients.copy()
    clients["fraud_percentage"] = clients["fraud_rate"] * 100
    left, right = st.columns(2)
    with left:
        figure = px.bar(
            clients,
            x="client_id",
            y="sample_count",
            color="partition",
            barmode="group",
            title="Training transactions per institution",
            labels={"client_id": "Institution", "sample_count": "Transactions"},
            color_discrete_sequence=["#32d6c5", "#5c9dff"],
        )
        figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch")
    with right:
        figure = px.bar(
            clients,
            x="client_id",
            y="fraud_percentage",
            color="partition",
            barmode="group",
            title="Fraud prevalence by institution",
            labels={"client_id": "Institution", "fraud_percentage": "Fraud cases (%)"},
            color_discrete_sequence=["#32d6c5", "#ff7187"],
        )
        figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch")
    st.dataframe(
        clients[["partition", "client_id", "sample_count", "fraud_count", "fraud_percentage"]].rename(
            columns={"partition": "Distribution", "client_id": "Institution", "sample_count": "Transactions", "fraud_count": "Frauds", "fraud_percentage": "Fraud %"}
        ),
        hide_index=True,
        width="stretch",
    )
    if metrics is not None:
        relevant = metrics.loc[metrics["method"].isin(["fedavg", "dp_fedavg", "secure_fedavg", "secure_dp_fedavg"])]
        figure = px.line(
            relevant,
            x="partition",
            y="pr_auc_average_precision",
            color="method",
            markers=True,
            title="Observed PR-AUC by client distribution",
            labels={"partition": "Partition", "pr_auc_average_precision": "PR-AUC (average precision)", "method": "Method"},
            color_discrete_map=COLORS,
        )
        figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch")
    rounds = _require_frame(data, "per_round_metrics", "Per-round metrics")
    if rounds is not None:
        figure = px.line(
            rounds,
            x="round",
            y="test_loss",
            color="method",
            line_dash="partition",
            markers=True,
            title="Recorded global test loss across rounds",
            labels={"round": "Communication round", "test_loss": "Test loss", "method": "Method", "partition": "Distribution"},
            color_discrete_map=COLORS,
        )
        figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch")
    st.caption(
        "The run used one seed, so differences are descriptive for this split only; they do not establish "
        "that one partition strategy generally improves performance."
    )


def render_architecture(data: dict[str, Any]) -> None:
    st.title("System architecture")
    st.caption("Illustrative data flow for the recorded, single-process research simulation.")
    nodes = {
        "Dataset": (0.08, 0.72, "Credit-card dataset\nLocal transaction records"),
        "Preprocessing": (0.27, 0.72, "Train-only scaling\nFixed feature order"),
        "Clients": (0.48, 0.72, "Five institutions\nPrivate local records"),
        "Local training": (0.69, 0.72, "FinancialFraudMLP\nLocal model updates"),
        "Privacy": (0.69, 0.38, "Optional DP-SGD\nOptional pairwise masks"),
        "Aggregation": (0.48, 0.38, "FedAvg or secure\nmasked aggregation"),
        "Global model": (0.27, 0.38, "Aggregated fraud\nclassifier"),
        "Evaluation": (0.08, 0.38, "Held-out test metrics\nPR-AUC / recall"),
    }
    figure = go.Figure()
    edge_pairs = [
        ("Dataset", "Preprocessing", "raw records"),
        ("Preprocessing", "Clients", "partitioned features"),
        ("Clients", "Local training", "local batches"),
        ("Local training", "Privacy", "local update"),
        ("Privacy", "Aggregation", "plain or masked update"),
        ("Aggregation", "Global model", "aggregate"),
        ("Global model", "Evaluation", "model only"),
    ]
    for source, target, label in edge_pairs:
        x0, y0, _ = nodes[source]
        x1, y1, _ = nodes[target]
        figure.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="#32d6c5",
            text=label,
            font={"color": "#cbd8e8", "size": 10},
            bgcolor="#101d2e",
        )
    for name, (x, y, description) in nodes.items():
        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                text=[f"<b>{name}</b><br>{description}"],
                textposition="middle center",
                textfont={"size": 12, "color": "#f2f7ff"},
                marker={"size": 52, "color": "#15304a", "line": {"color": "#32d6c5", "width": 2}},
                hovertemplate=f"<b>{name}</b><br>{description.replace(chr(10), '<br>')}<extra></extra>",
                showlegend=False,
            )
        )
    figure.update_layout(
        template="plotly_dark",
        height=580,
        xaxis={"visible": False, "range": [0, 0.82]},
        yaxis={"visible": False, "range": [0.18, 0.9]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(figure, width="stretch")
    st.markdown(
        """
        <div class="notice-grid">
          <div class="notice-card"><b>Raw records</b><br>Remain in client-local loaders during training. They are not uploaded to the aggregation server.</div>
          <div class="notice-card"><b>Model updates</b><br>FedAvg sends client model states; secure mode submits pairwise-masked states for aggregation.</div>
          <div class="notice-card"><b>Simulation boundary</b><br>Clients and server share one process here. Network authentication and process isolation are not implemented.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_experiment_results(data: dict[str, Any]) -> None:
    st.title("Experiment results")
    st.caption("Historical Phase 8 outputs. These artifacts are not regenerated by the dashboard.")
    metrics = _require_frame(data, "final_metrics", "Final metrics")
    if metrics is not None:
        st.subheader("Final metrics")
        st.dataframe(metrics, hide_index=True, width="stretch")
        st.download_button(
            "Download final metrics CSV",
            metrics.to_csv(index=False).encode("utf-8"),
            file_name="phase8_final_metrics.csv",
            mime="text/csv",
        )
    for key, label in (
        ("experiment_config", "Experiment configuration"),
        ("dataset_statistics", "Dataset statistics"),
        ("privacy_accounting", "Privacy accounting"),
        ("client_statistics", "Client fraud distribution"),
    ):
        value = data.get(key)
        if value is not None:
            with st.expander(label):
                st.json(value)
                st.download_button(
                    f"Download {label.lower()} JSON",
                    json.dumps(value, indent=2, default=str).encode("utf-8"),
                    file_name=f"phase8_{key}.json",
                    mime="application/json",
                    key=f"download_{key}",
                )
        else:
            _show_artifact_errors(data, [key])

    for key, label in (
        ("per_round_metrics", "Per-round metrics"),
        ("communication_overhead", "Communication overhead"),
        ("client_distribution", "Client statistics CSV"),
    ):
        frame = data.get(key)
        if isinstance(frame, pd.DataFrame):
            with st.expander(label):
                st.dataframe(frame, hide_index=True, width="stretch")
                st.download_button(
                    f"Download {label} CSV",
                    frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"phase8_{key}.csv",
                    mime="text/csv",
                    key=f"download_{key}",
                )
        else:
            _show_artifact_errors(data, [key])

    report = data.get("report")
    if isinstance(report, str):
        with st.expander("Phase 8 final report", expanded=True):
            st.markdown(report)
            st.download_button(
                "Download Phase 8 report",
                report.encode("utf-8"),
                file_name="phase8_final_report.md",
                mime="text/markdown",
            )
    else:
        _show_artifact_errors(data, ["report"])

    graphs = data.get("graphs", [])
    if graphs:
        st.subheader("Saved Phase 8 graphs")
        selected_graph = st.selectbox("Graph", graphs, format_func=lambda path: path.name)
        st.image(str(selected_graph), caption=selected_graph.name, width="stretch")
        graph_bytes = selected_graph.read_bytes()
        st.download_button(
            "Download selected graph",
            graph_bytes,
            file_name=selected_graph.name,
            mime="image/png",
        )
    else:
        _show_artifact_errors(data, ["graphs"])
