import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Advanced GNN Fraud Detection",
    page_icon="💳",
    layout="wide"
)

CHART_DIR = "backend_charts"

# =========================
# CSS + ANIMATIONS
# =========================
st.markdown("""
<style>
.stApp {
    background:
    radial-gradient(circle at top left, rgba(37,99,235,0.35), transparent 30%),
    radial-gradient(circle at top right, rgba(236,72,153,0.30), transparent 30%),
    linear-gradient(135deg, #020617, #0f172a, #1e1b4b, #4c1d95, #831843);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #111827, #312e81);
}

.hero {
    padding: 70px;
    border-radius: 40px;
    text-align: center;
    background:
    linear-gradient(135deg, rgba(37,99,235,0.82), rgba(147,51,234,0.82), rgba(219,39,119,0.78)),
    url("https://images.unsplash.com/photo-1642104704074-907c0698cbd9?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    box-shadow: 0 0 70px rgba(236,72,153,0.55);
    animation: heroZoom 1.3s ease-in-out;
}

.hero h1 {
    font-size: 64px;
    color: white;
}

.hero p {
    font-size: 23px;
    color: #f3f4f6;
}

.card {
    background: rgba(255,255,255,0.11);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.20);
    padding: 26px;
    border-radius: 28px;
    box-shadow: 0 0 35px rgba(59,130,246,0.25);
    animation: slideUp 0.9s ease;
    transition: all 0.35s ease;
}

.card:hover {
    transform: translateY(-8px) scale(1.01);
    box-shadow: 0 0 50px rgba(236,72,153,0.48);
}

.metric-box {
    padding: 26px;
    border-radius: 26px;
    text-align: center;
    background: linear-gradient(135deg, #2563eb, #7c3aed, #db2777);
    box-shadow: 0 0 32px rgba(124,58,237,0.60);
    animation: floatCard 3s ease-in-out infinite;
}

.risk-low {
    padding: 35px;
    border-radius: 28px;
    background: linear-gradient(135deg, #059669, #10b981);
    text-align: center;
}

.risk-mid {
    padding: 35px;
    border-radius: 28px;
    background: linear-gradient(135deg, #d97706, #f59e0b);
    text-align: center;
}

.risk-high {
    padding: 35px;
    border-radius: 28px;
    background: linear-gradient(135deg, #dc2626, #be123c);
    text-align: center;
}

.swipe-container {
    display: flex;
    overflow-x: auto;
    gap: 24px;
    padding: 22px 5px;
    scroll-behavior: smooth;
}

.swipe-card {
    min-width: 370px;
    background: rgba(255,255,255,0.12);
    border-radius: 28px;
    padding: 18px;
    border: 1px solid rgba(255,255,255,0.20);
    box-shadow: 0 0 30px rgba(59,130,246,0.28);
    transition: 0.35s ease;
}

.swipe-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 45px rgba(236,72,153,0.50);
}

.swipe-card img {
    width: 100%;
    height: 215px;
    object-fit: cover;
    border-radius: 22px;
}

.pipeline {
    text-align:center;
    padding:26px;
    border-radius:28px;
    background: linear-gradient(135deg, rgba(37,99,235,0.65), rgba(219,39,119,0.58));
    font-size:21px;
    box-shadow:0 0 38px rgba(124,58,237,0.50);
}

.badge {
    display:inline-block;
    padding:10px 16px;
    border-radius:999px;
    background:rgba(255,255,255,0.16);
    margin:6px;
    border:1px solid rgba(255,255,255,0.18);
}

@keyframes heroZoom {
    from {transform: scale(0.95); opacity: 0;}
    to {transform: scale(1); opacity: 1;}
}

@keyframes slideUp {
    from {transform: translateY(50px); opacity: 0;}
    to {transform: translateY(0); opacity: 1;}
}

@keyframes floatCard {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-6px);}
    100% {transform: translateY(0px);}
}
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def identify_files(uploaded_files):
    files = {}
    for f in uploaded_files:
        name = f.name.lower()
        if "feature" in name:
            files["features"] = f
        elif "edge" in name:
            files["edges"] = f
        elif "class" in name:
            files["classes"] = f
    return files


@st.cache_data
def load_datasets(features_file, edges_file, classes_file):
    features = pd.read_csv(features_file, header=None)
    edges = pd.read_csv(edges_file)
    classes = pd.read_csv(classes_file)

    feature_cols = ["txId", "time_step"] + [
        f"feature_{i}" for i in range(1, features.shape[1] - 1)
    ]

    features.columns = feature_cols
    df = features.merge(classes, on="txId", how="left")

    return features, edges, classes, df


def prepare_analysis_df(df):
    temp = df.copy()
    temp = temp[temp["class"] != "unknown"].copy()
    temp["class"] = temp["class"].astype(int)
    temp["label"] = temp["class"].map({2: "Normal", 1: "Fraud"})
    return temp


def compute_graph_degree(edges):
    degree_df = pd.DataFrame({
        "txId": pd.concat([edges["txId1"], edges["txId2"]])
    })
    return degree_df["txId"].value_counts().to_dict()


def calculate_risk(row, degree_map=None):
    feature_cols = [c for c in row.index if str(c).startswith("feature_")]
    vals = row[feature_cols].astype(float).values

    mean_risk = abs(vals.mean())
    std_risk = abs(vals.std())
    max_risk = abs(vals.max())
    min_risk = abs(vals.min())
    time_risk = int(row["time_step"]) / 49

    degree = 0
    if degree_map:
        degree = degree_map.get(row["txId"], 0)

    graph_risk = min(100, degree * 6)

    score = (
        mean_risk * 4 +
        std_risk * 6 +
        max_risk * 1.4 +
        min_risk * 1.1 +
        time_risk * 15 +
        graph_risk * 0.25
    )

    return int(min(100, max(0, score)))


def add_prediction_columns(df, degree_map=None):
    temp = df.copy()
    temp["fraud_probability"] = temp.apply(
        lambda row: calculate_risk(row, degree_map),
        axis=1
    )
    temp["prediction"] = temp["fraud_probability"].apply(
        lambda x: "Fraud" if x >= 70 else "Suspicious" if x >= 35 else "Normal"
    )
    return temp


def risk_status(score):
    if score < 35:
        return "LOW RISK", "risk-low"
    elif score < 70:
        return "MEDIUM RISK", "risk-mid"
    else:
        return "HIGH RISK", "risk-high"


def show_chart_file(title, filename, description=""):
    path = os.path.join(CHART_DIR, filename)

    st.markdown(f"""
    <div class="card">
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"Missing chart: {filename}")


def feature_gallery():
    st.markdown("""
    <div class="swipe-container">
        <div class="swipe-card">
            <img src="https://images.unsplash.com/photo-1639322537228-f710d846310a?auto=format&fit=crop&w=900&q=80">
            <h3>Blockchain Graphs</h3>
            <p>Explore transaction connections and network behavior.</p>
        </div>
        <div class="swipe-card">
            <img src="https://images.unsplash.com/photo-1563013544-824ae1b704d3?auto=format&fit=crop&w=900&q=80">
            <h3>Fraud Risk Scoring</h3>
            <p>Classify transactions into normal, suspicious and fraud.</p>
        </div>
        <div class="swipe-card">
            <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=900&q=80">
            <h3>Visual Analytics</h3>
            <p>Use charts, curves, heatmaps and 3D graphs.</p>
        </div>
        <div class="swipe-card">
            <img src="https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&w=900&q=80">
            <h3>Anomaly Detection</h3>
            <p>Find abnormal transactions using Isolation Forest.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================
# SESSION STATE
# =========================
for key in ["features", "edges", "classes", "df", "degree_map"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# SIDEBAR
# =========================
st.sidebar.title("💳 GNN Fraud App")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Upload",
        "Dashboard",
        "Prediction",
        "Anomaly Detection",
        "Model Performance",
        "Project Info"
    ]
)

# =========================
# HOME
# =========================
if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>Advanced Financial Fraud Detection</h1>
        <p>Graph-based AI dashboard for cryptocurrency fraud intelligence.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Graph AI", "Transaction network"),
        ("Risk Engine", "Fraud scoring"),
        ("Model Curves", "Accuracy + ROC"),
        ("Reports", "Download outputs")
    ]

    for col, item in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <h2>{item[0]}</h2>
                <p>{item[1]}</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="pipeline">
    Upload Dataset → Validate Data → Analyze Graph → Predict Fraud → Detect Anomalies → View Model Curves → Download Report
    </div>
    """, unsafe_allow_html=True)

    st.subheader("🌈 Feature Gallery")
    feature_gallery()

# =========================
# DATA UPLOAD
# =========================
elif page == "Data Upload":

    st.subheader("📁 Upload Dataset Files")

    uploaded_files = st.file_uploader(
        "Upload features, edgelist, and classes CSV files together",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        files = identify_files(uploaded_files)

        if len(files) == 3:
            with st.spinner("Loading datasets and preparing graph metadata..."):
                features, edges, classes, df = load_datasets(
                    files["features"],
                    files["edges"],
                    files["classes"]
                )
                degree_map = compute_graph_degree(edges)

            st.session_state.features = features
            st.session_state.edges = edges
            st.session_state.classes = classes
            st.session_state.df = df
            st.session_state.degree_map = degree_map

            st.success("Datasets loaded successfully!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transactions", features.shape[0])
            c2.metric("Features", features.shape[1])
            c3.metric("Edges", edges.shape[0])
            c4.metric("Labels", classes.shape[0])

            st.subheader("🩺 Dataset Health Check")

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Missing Values", int(df.isnull().sum().sum()))
            h2.metric("Duplicate Rows", int(df.duplicated().sum()))
            h3.metric("Unknown Labels", int((df["class"] == "unknown").sum()))
            h4.metric("Graph Nodes", len(degree_map))

            st.dataframe(df.head(20), use_container_width=True)
        else:
            st.error("Please upload all 3 files: features, edgelist, and classes.")

# =========================
# DASHBOARD
# =========================
elif page == "Dashboard":

    st.subheader("📊 Combined Data, Graph and Risk Dashboard")

    if st.session_state.df is None:
        st.warning("Upload datasets first.")
    else:
        df = st.session_state.df
        edges = st.session_state.edges

        analysis_df = prepare_analysis_df(df)
        pred_df = add_prediction_columns(
            analysis_df.sample(min(8000, len(analysis_df)), random_state=42),
            st.session_state.degree_map
        )

        tab1, tab2, tab3 = st.tabs(
            ["Dataset Overview", "Risk Analytics", "Graph Intelligence"]
        )

        with tab1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Transactions", df.shape[0])
            c2.metric("Labeled", analysis_df.shape[0])
            c3.metric("Fraud", int((analysis_df["label"] == "Fraud").sum()))
            c4.metric("Normal", int((analysis_df["label"] == "Normal").sum()))

            col1, col2 = st.columns(2)

            label_counts = analysis_df["label"].value_counts().reset_index()
            label_counts.columns = ["Class", "Count"]

            with col1:
                fig = px.pie(
                    label_counts,
                    names="Class",
                    values="Count",
                    hole=0.45,
                    title="Fraud vs Normal Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig2 = px.histogram(
                    analysis_df,
                    x="time_step",
                    color="label",
                    barmode="group",
                    title="Transactions by Time Step"
                )
                st.plotly_chart(fig2, use_container_width=True)

            numeric_df = analysis_df.select_dtypes(include=np.number).iloc[:, :35]
            fig3 = px.imshow(
                numeric_df.corr(),
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig3, use_container_width=True)

        with tab2:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Analyzed", len(pred_df))
            c2.metric("High Risk", int((pred_df["prediction"] == "Fraud").sum()))
            c3.metric("Suspicious", int((pred_df["prediction"] == "Suspicious").sum()))
            c4.metric("Average Risk", round(pred_df["fraud_probability"].mean(), 2))

            col1, col2 = st.columns(2)

            with col1:
                fig4 = px.pie(
                    pred_df,
                    names="prediction",
                    hole=0.45,
                    title="Risk Distribution"
                )
                st.plotly_chart(fig4, use_container_width=True)

            with col2:
                fig5 = px.histogram(
                    pred_df,
                    x="fraud_probability",
                    color="prediction",
                    nbins=45,
                    title="Risk Score Distribution"
                )
                st.plotly_chart(fig5, use_container_width=True)

            timeline = pred_df.groupby(["time_step", "prediction"]).size().reset_index(name="count")
            fig6 = px.area(
                timeline,
                x="time_step",
                y="count",
                color="prediction",
                title="Risk Trend Over Time"
            )
            st.plotly_chart(fig6, use_container_width=True)

            fig7 = px.scatter_3d(
                pred_df,
                x="time_step",
                y="fraud_probability",
                z="class",
                color="prediction",
                size="fraud_probability",
                hover_data=["txId", "label"],
                title="3D Risk Space"
            )
            st.plotly_chart(fig7, use_container_width=True)

        with tab3:
            total_nodes = len(set(edges["txId1"]).union(set(edges["txId2"])))
            total_edges = edges.shape[0]
            avg_degree = round((2 * total_edges) / total_nodes, 2)

            g1, g2, g3 = st.columns(3)
            g1.metric("Total Nodes", total_nodes)
            g2.metric("Total Edges", total_edges)
            g3.metric("Average Degree", avg_degree)

            degree_df = pd.DataFrame({
                "Transaction": pd.concat([edges["txId1"], edges["txId2"]])
            })

            degree_count = degree_df["Transaction"].value_counts().head(30).reset_index()
            degree_count.columns = ["Transaction", "Degree"]

            fig8 = px.bar(
                degree_count,
                x="Transaction",
                y="Degree",
                title="Top High-Degree Transactions"
            )
            st.plotly_chart(fig8, use_container_width=True)

            sample_edges = edges.head(140)
            G = nx.from_pandas_edgelist(sample_edges, "txId1", "txId2")
            pos = nx.spring_layout(G, seed=42)

            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            node_x, node_y, node_text = [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1),
                hoverinfo="none"
            )

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hovertext=node_text,
                marker=dict(
                    size=11,
                    color=list(range(len(node_x))),
                    colorscale="Plasma",
                    showscale=True
                )
            )

            fig9 = go.Figure(data=[edge_trace, node_trace])
            fig9.update_layout(
                title="Interactive Transaction Network",
                showlegend=False
            )
            st.plotly_chart(fig9, use_container_width=True)

# =========================
# PREDICTION
# =========================
elif page == "Prediction":

    st.subheader("🔍 Single and Batch Prediction")

    if st.session_state.df is None:
        st.warning("Upload datasets first.")
    else:
        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

        df = prepare_analysis_df(st.session_state.df)
        predicted_df = add_prediction_columns(df, st.session_state.degree_map)

        with tab1:
            risk_filter = st.radio(
                "Select risk category",
                ["Random", "Low Risk", "Medium Risk", "High Risk"],
                horizontal=True
            )

            if risk_filter == "Low Risk":
                filtered = predicted_df[predicted_df["fraud_probability"] < 35]
            elif risk_filter == "Medium Risk":
                filtered = predicted_df[
                    (predicted_df["fraud_probability"] >= 35) &
                    (predicted_df["fraud_probability"] < 70)
                ]
            elif risk_filter == "High Risk":
                filtered = predicted_df[predicted_df["fraud_probability"] >= 70]
            else:
                filtered = predicted_df.sample(min(5000, len(predicted_df)))

            if len(filtered) == 0:
                st.warning("No transaction found for this category.")
            else:
                selected_tx = st.selectbox(
                    "Select Transaction ID",
                    filtered["txId"].head(5000).tolist()
                )

                row = filtered[filtered["txId"] == selected_tx].iloc[0]
                score = int(row["fraud_probability"])
                label, css = risk_status(score)

                st.markdown(f"""
                <div class="{css}">
                    <h1>{label}</h1>
                    <h2>Fraud Probability: {score}%</h2>
                </div>
                """, unsafe_allow_html=True)

                degree = st.session_state.degree_map.get(row["txId"], 0)

                factors = pd.DataFrame({
                    "Factor": ["Mean Risk", "Std Risk", "Max Risk", "Time Risk", "Graph Degree"],
                    "Score": [
                        min(100, abs(row.filter(like="feature_").astype(float).mean()) * 8),
                        min(100, abs(row.filter(like="feature_").astype(float).std()) * 8),
                        min(100, abs(row.filter(like="feature_").astype(float).max()) * 5),
                        min(100, int(row["time_step"]) * 5),
                        min(100, degree * 6)
                    ]
                })

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        factors,
                        x="Factor",
                        y="Score",
                        text="Score",
                        title="Risk Factor Breakdown"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        title={"text": "Fraud Risk Gauge"},
                        gauge={"axis": {"range": [0, 100]}}
                    ))
                    st.plotly_chart(gauge, use_container_width=True)

                radar = go.Figure()
                radar.add_trace(go.Scatterpolar(
                    r=factors["Score"],
                    theta=factors["Factor"],
                    fill="toself"
                ))
                radar.update_layout(
                    title="Risk Radar",
                    polar=dict(radialaxis=dict(visible=True, range=[0,100]))
                )
                st.plotly_chart(radar, use_container_width=True)

                st.dataframe(row.to_frame().T, use_container_width=True)

        with tab2:
            sample_size = st.slider(
                "Transactions to analyze",
                100,
                min(15000, len(df)),
                1500
            )

            batch = df.sample(sample_size, random_state=42).copy()
            batch = add_prediction_columns(batch, st.session_state.degree_map)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Analyzed", len(batch))
            c2.metric("Fraud", int((batch["prediction"] == "Fraud").sum()))
            c3.metric("Suspicious", int((batch["prediction"] == "Suspicious").sum()))
            c4.metric("Normal", int((batch["prediction"] == "Normal").sum()))

            fig1 = px.scatter(
                batch,
                x="time_step",
                y="fraud_probability",
                color="prediction",
                size="fraud_probability",
                hover_data=["txId", "label"],
                title="Batch Transaction Risk Scatter"
            )
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = px.box(
                batch,
                x="prediction",
                y="fraud_probability",
                color="prediction",
                title="Risk Spread by Prediction"
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("🚨 Top Suspicious Transactions")
            top_suspicious = batch.sort_values(
                "fraud_probability",
                ascending=False
            ).head(20)

            st.dataframe(
                top_suspicious[["txId", "time_step", "label", "fraud_probability", "prediction"]],
                use_container_width=True
            )

            csv = batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Report",
                csv,
                "fraud_prediction_report.csv",
                "text/csv"
            )

# =========================
# ANOMALY DETECTION
# =========================
elif page == "Anomaly Detection":

    st.subheader("🚨 Advanced Anomaly Detection")

    if st.session_state.df is None:
        st.warning("Upload datasets first.")
    else:
        df = prepare_analysis_df(st.session_state.df).copy()
        feature_cols = [c for c in df.columns if str(c).startswith("feature_")]

        sample_size = st.slider(
            "Sample size",
            1000,
            min(12000, len(df)),
            6000
        )

        contamination = st.slider(
            "Anomaly sensitivity",
            0.01,
            0.20,
            0.08
        )

        sample = df.sample(sample_size, random_state=21).copy()

        X = sample[feature_cols].astype(float).values
        X_scaled = StandardScaler().fit_transform(X)

        iso = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        sample["anomaly_flag"] = iso.fit_predict(X_scaled)
        sample["anomaly_score"] = -iso.decision_function(X_scaled)

        sample["anomaly_level"] = pd.qcut(
            sample["anomaly_score"],
            q=3,
            labels=["Low", "Medium", "High"]
        )

        sample["fraud_probability"] = sample.apply(
            lambda row: calculate_risk(row, st.session_state.degree_map),
            axis=1
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analyzed", len(sample))
        c2.metric("Detected Anomalies", int((sample["anomaly_flag"] == -1).sum()))
        c3.metric("High Anomaly", int((sample["anomaly_level"] == "High").sum()))
        c4.metric("Sensitivity", contamination)

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(
                sample,
                x="anomaly_score",
                color="anomaly_level",
                nbins=55,
                title="Anomaly Score Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.pie(
                sample,
                names="anomaly_level",
                title="Anomaly Level Distribution",
                hole=0.45
            )
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter(
            sample,
            x="time_step",
            y="anomaly_score",
            color="anomaly_level",
            size="fraud_probability",
            hover_data=["txId", "label"],
            title="Anomaly Score Over Time"
        )
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter_3d(
            sample,
            x="time_step",
            y="anomaly_score",
            z="fraud_probability",
            color="anomaly_level",
            hover_data=["txId", "label"],
            title="3D Anomaly-Fraud Risk Space"
        )
        st.plotly_chart(fig4, use_container_width=True)

        fig5 = px.density_heatmap(
            sample,
            x="time_step",
            y="anomaly_score",
            nbinsx=30,
            nbinsy=30,
            title="Anomaly Density Heatmap"
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("🚨 Top Anomalous Transactions")
        top_anomaly = sample.sort_values(
            "anomaly_score",
            ascending=False
        ).head(25)

        st.dataframe(
            top_anomaly[["txId", "time_step", "label", "anomaly_score", "anomaly_level", "fraud_probability"]],
            use_container_width=True
        )

# =========================
# MODEL PERFORMANCE
# =========================
elif page == "Model Performance":

    st.subheader("📈 Model Performance and Graph Curves")

    history_path = os.path.join(CHART_DIR, "training_history.csv")
    report_path = os.path.join(CHART_DIR, "classification_report.txt")

    tab1, tab2, tab3 = st.tabs(
        ["Metric Curves", "Saved Graph Gallery", "Classification Report"]
    )

    with tab1:
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            latest = history_df.iloc[-1]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Final Loss", round(latest.get("loss", 0), 4))
            c2.metric("Accuracy", round(latest.get("val_accuracy", 0), 4))
            c3.metric("Precision", round(latest.get("val_precision", 0), 4))
            c4.metric("Recall", round(latest.get("val_recall", 0), 4))
            c5.metric("F1 Score", round(latest.get("val_f1", 0), 4))

            fig = px.line(
                history_df,
                x="epoch",
                y=["loss", "val_accuracy", "val_precision", "val_recall", "val_f1"],
                title="Training History: Loss, Accuracy, Precision, Recall and F1"
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                fig_acc = px.line(
                    history_df,
                    x="epoch",
                    y="val_accuracy",
                    title="Validation Accuracy Curve"
                )
                st.plotly_chart(fig_acc, use_container_width=True)

            with col2:
                fig_loss = px.line(
                    history_df,
                    x="epoch",
                    y="loss",
                    title="Training Loss Curve"
                )
                st.plotly_chart(fig_loss, use_container_width=True)

            fig_metrics = px.line(
                history_df,
                x="epoch",
                y=["val_precision", "val_recall", "val_f1"],
                title="Precision, Recall and F1 Curve"
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        else:
            st.warning("training_history.csv missing from backend_charts folder.")

    with tab2:
        chart_groups = {
            "Training Curves": [
                ("Training Loss Curve", "loss_curve.png", "Shows how model loss decreases during training."),
                ("Validation Accuracy Curve", "accuracy_curve.png", "Shows model accuracy improvement over epochs."),
                ("Precision Recall F1 Curve", "precision_recall_f1_curve.png", "Shows precision, recall and F1 behavior.")
            ],
            "Evaluation Curves": [
                ("ROC Curve", "roc_curve.png", "Shows classifier separation ability using AUC."),
                ("Precision Recall Curve", "precision_recall_curve.png", "Important for imbalanced fraud detection.")
            ],
            "Heatmaps": [
                ("Confusion Matrix Heatmap", "confusion_matrix_heatmap.png", "Shows correct and incorrect predictions."),
                ("Normalized Confusion Matrix", "normalized_confusion_matrix_heatmap.png", "Shows prediction ratios."),
                ("Feature Correlation Heatmap", "feature_correlation_heatmap.png", "Shows feature relationship patterns.")
            ],
            "Distribution Charts": [
                ("Class Distribution", "class_distribution.png", "Shows normal vs fraud transaction balance."),
                ("Fraud Probability Distribution", "fraud_probability_distribution.png", "Shows predicted fraud probability spread.")
            ]
        }

        selected_group = st.selectbox(
            "Select chart group",
            list(chart_groups.keys())
        )

        for title, filename, desc in chart_groups[selected_group]:
            show_chart_file(title, filename, desc)

        st.subheader("Full Chart Gallery")

        all_charts = []
        for group in chart_groups.values():
            all_charts.extend(group)

        cols = st.columns(2)

        for i, (title, filename, desc) in enumerate(all_charts):
            with cols[i % 2]:
                show_chart_file(title, filename, desc)

    with tab3:
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                st.code(f.read())
        else:
            st.warning("classification_report.txt missing from backend_charts folder.")

# =========================
# PROJECT INFO
# =========================
elif page == "Project Info":

    st.subheader("📌 Project Information")

    tab1, tab2, tab3 = st.tabs(
        ["Overview", "Architecture", "Evaluation Summary"]
    )

    with tab1:
        st.markdown("""
        <div class="card">
            <h2>Graph-Based Financial Fraud Detection using Graph Neural Networks</h2>
            <p>
            This project detects suspicious cryptocurrency transactions using graph-based financial intelligence,
            anomaly detection, dynamic fraud risk scoring, and full model-performance visualization.
            </p>
        </div>
        """, unsafe_allow_html=True)

        feature_gallery()

    with tab2:
        st.markdown("""
        <div class="pipeline">
        Dataset Upload → Feature Processing → Graph Analysis → GNN Training → Model Evaluation → Dashboard Deployment
        </div>
        """, unsafe_allow_html=True)

        flow = pd.DataFrame({
            "Stage": ["Upload", "Clean", "Graph", "Train", "Evaluate", "Deploy"],
            "Importance": [90, 85, 95, 94, 96, 88]
        })

        fig = px.bar(
            flow,
            x="Stage",
            y="Importance",
            text="Importance",
            title="Project Architecture Flow"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        components.html(
            """
            <style>
            body {
                background: transparent;
                font-family: Arial, sans-serif;
                color: white;
            }

            .eval-title {
                padding: 28px;
                border-radius: 25px;
                background: linear-gradient(135deg, rgba(37,99,235,0.45), rgba(219,39,119,0.35));
                border: 1px solid rgba(255,255,255,0.22);
                box-shadow: 0 0 30px rgba(124,58,237,0.35);
                margin-bottom: 25px;
            }

            .eval-title h1 {
                margin: 0;
                font-size: 34px;
                color: white;
            }

            .eval-title p {
                color: #e5e7eb;
                font-size: 16px;
            }

            .eval-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 22px;
            }

            .eval-card {
                min-height: 210px;
                padding: 24px;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(30,64,175,0.88), rgba(126,34,206,0.82), rgba(190,24,93,0.72));
                border: 1px solid rgba(255,255,255,0.20);
                box-shadow: 0 0 25px rgba(59,130,246,0.30);
                transition: all 0.4s ease;
                overflow: hidden;
            }

            .eval-card:hover {
                transform: translateY(-10px) scale(1.03);
                box-shadow: 0 0 45px rgba(236,72,153,0.65);
            }

            .eval-icon {
                font-size: 36px;
                margin-bottom: 8px;
            }

            .eval-card h3 {
                margin: 8px 0;
                font-size: 24px;
                color: white;
            }

            .short {
                font-size: 15px;
                color: #f1f5f9;
            }

            .details {
                opacity: 0;
                transform: translateY(20px);
                transition: all 0.4s ease;
                font-size: 14px;
                line-height: 1.55;
                color: #ffffff;
            }

            .eval-card:hover .details {
                opacity: 1;
                transform: translateY(0);
            }
            </style>

            <div class="eval-title">
                <h1>📊 Evaluation Summary</h1>
                <p>Hover over each card to understand every model evaluation metric.</p>
            </div>

            <div class="eval-grid">

                <div class="eval-card">
                    <div class="eval-icon">📉</div>
                    <h3>Loss Curve</h3>
                    <p class="short">Tracks model learning error.</p>
                    <p class="details">A decreasing loss curve means the GraphSAGE model is learning properly. If loss stays high, the model may be underfitting.</p>
                </div>

                <div class="eval-card">
                    <div class="eval-icon">✅</div>
                    <h3>Accuracy</h3>
                    <p class="short">Measures total correct predictions.</p>
                    <p class="details">Accuracy shows overall correctness, but in fraud detection it should not be used alone because fraud data is usually imbalanced.</p>
                </div>

                <div class="eval-card">
                    <div class="eval-icon">🎯</div>
                    <h3>Precision</h3>
                    <p class="short">Measures correct fraud alerts.</p>
                    <p class="details">High precision means most transactions predicted as fraud are actually fraud. It helps reduce false alarms.</p>
                </div>

                <div class="eval-card">
                    <div class="eval-icon">🔍</div>
                    <h3>Recall</h3>
                    <p class="short">Measures fraud detection coverage.</p>
                    <p class="details">Recall shows how many real fraud transactions were detected. It is very important because missing fraud can be costly.</p>
                </div>

                <div class="eval-card">
                    <div class="eval-icon">⚖️</div>
                    <h3>F1 Score</h3>
                    <p class="short">Balances precision and recall.</p>
                    <p class="details">F1 score is useful for imbalanced fraud datasets because it balances false alarms and missed fraud cases.</p>
                </div>

                <div class="eval-card">
                    <div class="eval-icon">📈</div>
                    <h3>ROC Curve</h3>
                    <p class="short">Shows class separation ability.</p>
                    <p class="details">ROC curve shows how well the model separates fraud and normal transactions. Higher AUC means stronger performance.</p>
                </div>

                <div class="eval-card">
                    <div class="eval-icon">📊</div>
                    <h3>Precision-Recall Curve</h3>
                    <p class="short">Best for imbalanced data.</p>
                    <p class="details">This curve is very useful when fraud cases are rare. It shows the trade-off between catching fraud and avoiding false positives.</p>
                </div>

                <div class="eval-card">
                    <div class="eval-icon">🔥</div>
                    <h3>Confusion Matrix</h3>
                    <p class="short">Shows prediction breakdown.</p>
                    <p class="details">It shows true normal, true fraud, false alarms, and missed fraud cases. This helps identify model mistakes clearly.</p>
                </div>

                <div class="eval-card">
                    <div class="eval-icon">🧪</div>
                    <h3>Classification Report</h3>
                    <p class="short">Complete metric summary.</p>
                    <p class="details">It includes precision, recall, F1-score, and support for both fraud and normal classes.</p>
                </div>

            </div>
            """,
            height=900,
            scrolling=True
        )
