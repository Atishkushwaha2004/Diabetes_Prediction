import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Diabates.joblib")

model = load_model()

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a4e 50%, #0f0c29 100%);
    color: #e2e8f0;
}

/* ── Hide default header ── */
header[data-testid="stHeader"] { background: transparent; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(102,126,234,0.4);
}
.hero h1 { font-size: 2.8rem; font-weight: 700; color: white; margin: 0; letter-spacing: -0.5px; }
.hero p  { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin-top: 0.5rem; }

/* ── Metric cards ── */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 35px rgba(102,126,234,.3); }
.metric-card .label { font-size: .75rem; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-bottom: .4rem; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #818cf8; }
.metric-card .sub   { font-size: .8rem; color: #64748b; margin-top: .2rem; }

/* ── Result card ── */
.result-positive {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 2px solid rgba(239,68,68,0.5);
    border-radius: 18px; padding: 1.8rem; text-align: center;
    box-shadow: 0 8px 32px rgba(239,68,68,0.2);
}
.result-negative {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
    border: 2px solid rgba(34,197,94,0.5);
    border-radius: 18px; padding: 1.8rem; text-align: center;
    box-shadow: 0 8px 32px rgba(34,197,94,0.2);
}
.result-positive h2 { color: #f87171; font-size: 1.8rem; margin: 0; }
.result-negative h2 { color: #4ade80; font-size: 1.8rem; margin: 0; }

/* ── Section title ── */
.section-title {
    font-size: 1.3rem; font-weight: 700; color: #818cf8;
    border-left: 4px solid #818cf8; padding-left: .8rem; margin: 1.5rem 0 1rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%) !important;
    border-right: 1px solid rgba(129,140,248,0.2);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSlider > div > div > div { background: #818cf8 !important; }

/* ── Slider labels ── */
.stSlider label { color: #c4b5fd !important; font-weight: 600; }

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important;
    border-radius: 14px !important; padding: 0.8rem 2rem !important;
    font-size: 1.05rem !important; font-weight: 600 !important;
    letter-spacing: .3px !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.45) !important;
    transition: all .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(102,126,234,.6) !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.04); border-radius: 12px; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; border-radius: 10px !important; }
.stTabs [aria-selected="true"] { background: rgba(129,140,248,0.25) !important; color: #818cf8 !important; }

/* ── Risk badge ── */
.badge {
    display: inline-block; border-radius: 999px; padding: .25rem .9rem;
    font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
}
.badge-green { background: rgba(34,197,94,.2); color: #4ade80; border: 1px solid #4ade80; }
.badge-yellow{ background: rgba(234,179,8,.2);  color: #facc15; border: 1px solid #facc15; }
.badge-red   { background: rgba(239,68,68,.2);  color: #f87171; border: 1px solid #f87171; }

/* ── Info block ── */
.info-block {
    background: rgba(129,140,248,0.08);
    border: 1px solid rgba(129,140,248,0.2);
    border-radius: 12px; padding: 1rem 1.2rem; margin: .6rem 0;
}
.info-block b { color: #c4b5fd; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🩺 Diabetes AI Predictor</h1>
  <p>Powered by Gaussian Naïve Bayes · Pima Indians Diabetes Dataset</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL STATS (top row)
# ═══════════════════════════════════════════════════════════════════════════
total_samples = int(model.class_count_.sum())
diabetic_count = int(model.class_count_[1])
non_diabetic_count = int(model.class_count_[0])
prevalence = model.class_prior_[1] * 100

c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub in [
    (c1, "Training Samples", total_samples, "PIMA dataset"),
    (c2, "Diabetic Cases",  diabetic_count, f"{model.class_prior_[1]*100:.1f}% of total"),
    (c3, "Non-Diabetic",    non_diabetic_count, f"{model.class_prior_[0]*100:.1f}% of total"),
    (c4, "Features Used",   model.n_features_in_, "Clinical biomarkers"),
]:
    col.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{val}</div>
      <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR — input sliders
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎛️ Patient Parameters")
    st.markdown("---")

    pregnancies = st.slider("🤰 Pregnancies", 0, 17, 3,
        help="Number of times pregnant")
    glucose = st.slider("🍬 Glucose (mg/dL)", 0, 200, 120,
        help="Plasma glucose concentration (2-hour OGTT)")
    bp = st.slider("💓 Blood Pressure (mm Hg)", 0, 122, 70,
        help="Diastolic blood pressure")
    skin = st.slider("📏 Skin Thickness (mm)", 0, 99, 20,
        help="Triceps skin fold thickness")
    insulin = st.slider("💉 Insulin (μU/mL)", 0, 846, 80,
        help="2-Hour serum insulin")
    bmi = st.slider("⚖️ BMI (kg/m²)", 0.0, 67.1, 28.0, step=0.1,
        help="Body Mass Index")
    dpf = st.slider("🧬 Diabetes Pedigree Function", 0.078, 2.42, 0.47, step=0.001,
        help="Diabetes mellitus hereditary risk score")
    age = st.slider("🎂 Age (years)", 21, 81, 35,
        help="Patient age in years")

    st.markdown("---")
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True)

    st.markdown("""
    <div class="info-block" style="margin-top:1rem">
    <b>Model:</b> Gaussian Naïve Bayes<br>
    <b>Library:</b> scikit-learn<br>
    <b>Dataset:</b> Pima Indians Diabetes
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Feature Analysis", "🧠 Model Insights"])

features = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']
user_vals = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
X = np.array([user_vals])

with tab1:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        prediction   = model.predict(X)[0]
        proba        = model.predict_proba(X)[0]
        diabetic_prob = proba[1] * 100
        safe_prob     = proba[0] * 100

        if predict_btn or True:   # always show live result
            if prediction == 1:
                risk_label = "HIGH RISK" if diabetic_prob > 70 else "MODERATE RISK"
                badge_cls  = "badge-red" if diabetic_prob > 70 else "badge-yellow"
                st.markdown(f"""
                <div class="result-positive">
                  <div style="font-size:3rem">⚠️</div>
                  <h2>Diabetic — Positive</h2>
                  <p style="color:#fca5a5;margin:.4rem 0">High likelihood of diabetes detected</p>
                  <span class="badge {badge_cls}">{risk_label}</span>
                </div>""", unsafe_allow_html=True)
            else:
                risk_label = "LOW RISK"
                badge_cls  = "badge-green"
                st.markdown(f"""
                <div class="result-negative">
                  <div style="font-size:3rem">✅</div>
                  <h2>Non-Diabetic — Negative</h2>
                  <p style="color:#86efac;margin:.4rem 0">No significant diabetes indicators</p>
                  <span class="badge {badge_cls}">{risk_label}</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=diabetic_prob,
            title={"text": "Diabetes Probability (%)", "font": {"color": "#e2e8f0", "size": 15}},
            number={"suffix": "%", "font": {"color": "#818cf8", "size": 36}},
            delta={"reference": 50, "increasing": {"color": "#f87171"}, "decreasing": {"color": "#4ade80"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#94a3b8"}},
                "bar":  {"color": "#818cf8"},
                "bgcolor": "rgba(255,255,255,0.05)",
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 35],  "color": "rgba(34,197,94,0.15)"},
                    {"range": [35, 65], "color": "rgba(234,179,8,0.15)"},
                    {"range": [65, 100],"color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {"line": {"color": "#f87171", "width": 3}, "value": 50},
            }
        ))
        fig_gauge.update_layout(
            height=260, margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0"
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Probability Breakdown</div>', unsafe_allow_html=True)

        fig_bar = go.Figure([
            go.Bar(
                x=["Non-Diabetic", "Diabetic"],
                y=[safe_prob, diabetic_prob],
                marker_color=["rgba(34,197,94,0.7)", "rgba(239,68,68,0.7)"],
                marker_line=dict(color=["#4ade80","#f87171"], width=2),
                text=[f"{safe_prob:.1f}%", f"{diabetic_prob:.1f}%"],
                textposition="outside",
                textfont=dict(color="#e2e8f0", size=14, family="Inter"),
            )
        ])
        fig_bar.update_layout(
            height=280, margin=dict(t=20, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", yaxis=dict(range=[0,115], showgrid=False, zeroline=False),
            xaxis=dict(showgrid=False), showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Patient summary table
        st.markdown('<div class="section-title">Patient Summary</div>', unsafe_allow_html=True)
        icons = ["🤰","🍬","💓","📏","💉","⚖️","🧬","🎂"]
        means_no_diabetes = model.theta_[0]
        means_diabetes    = model.theta_[1]

        for i, (feat, icon, val, mn, md) in enumerate(
            zip(features, icons, user_vals, means_no_diabetes, means_diabetes)):
            # simple color hint
            diff_d = abs(val - md)
            diff_n = abs(val - mn)
            closer = "🔴" if diff_d < diff_n else "🟢"
            st.markdown(f"""
            <div class="info-block">
            <b>{icon} {feat}</b> &nbsp;
            <span style="color:#818cf8;font-size:1.1rem;font-weight:700">{val}</span>
            &nbsp;{closer}&nbsp;
            <span style="color:#64748b;font-size:.8rem">
              (Non-DM avg: {mn:.1f} | DM avg: {md:.1f})
            </span>
            </div>""", unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="section-title">Your Values vs. Population Averages</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    # Radar chart
    with col_a:
        # Normalize to 0-1 for radar
        feat_mins  = [0,    0,   0,  0,   0,   0,    0.078, 21]
        feat_maxes = [17, 200, 122, 99, 846, 67.1, 2.42,  81]
        def norm(vals):
            return [(v-mn)/(mx-mn) for v,mn,mx in zip(vals, feat_mins, feat_maxes)]

        user_n  = norm(user_vals)
        mean_d  = norm(model.theta_[1].tolist())
        mean_nd = norm(model.theta_[0].tolist())

        radar_labels = features + [features[0]]
        fig_radar = go.Figure()
        for vals, name, color, fcolor in [
            (user_n  + [user_n[0]],  "You",         "#818cf8", "rgba(129,140,248,0.12)"),
            (mean_d  + [mean_d[0]],  "Diabetic avg","#f87171", "rgba(248,113,113,0.12)"),
            (mean_nd + [mean_nd[0]], "Non-DM avg",  "#4ade80", "rgba(74,222,128,0.12)"),
        ]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=radar_labels, fill='toself',
                name=name, line=dict(color=color, width=2),
                fillcolor=fcolor,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], tickfont=dict(color="#64748b")),
                angularaxis=dict(tickfont=dict(color="#94a3b8")),
            ),
            height=360, margin=dict(t=20,b=20,l=20,r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Bar comparison
    with col_b:
        fig_comp = go.Figure()
        x_idx = list(range(len(features)))
        fig_comp.add_trace(go.Bar(
            name="Your Values", x=features, y=user_vals,
            marker_color="rgba(129,140,248,0.75)",
            marker_line=dict(color="#818cf8",width=1.5),
        ))
        fig_comp.add_trace(go.Scatter(
            name="Diabetic Avg", x=features, y=model.theta_[1],
            mode="lines+markers", line=dict(color="#f87171",width=2,dash="dot"),
            marker=dict(size=7, symbol="diamond"),
        ))
        fig_comp.add_trace(go.Scatter(
            name="Non-DM Avg", x=features, y=model.theta_[0],
            mode="lines+markers", line=dict(color="#4ade80",width=2,dash="dot"),
            marker=dict(size=7, symbol="circle"),
        ))
        fig_comp.update_layout(
            height=360, margin=dict(t=20,b=40,l=20,r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", barmode="overlay",
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Feature deviation heatmap row
    st.markdown('<div class="section-title">Deviation from Population Mean</div>', unsafe_allow_html=True)
    deviations = [(v - md) for v, md in zip(user_vals, model.theta_[1])]
    fig_dev = go.Figure(go.Bar(
        x=features, y=deviations,
        marker_color=["rgba(239,68,68,0.7)" if d > 0 else "rgba(34,197,94,0.7)" for d in deviations],
        marker_line=dict(color=["#f87171" if d>0 else "#4ade80" for d in deviations], width=1.5),
        text=[f"{d:+.1f}" for d in deviations], textposition="outside",
        textfont=dict(color="#e2e8f0"),
    ))
    fig_dev.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig_dev.update_layout(
        height=280, margin=dict(t=20,b=30,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    )
    st.plotly_chart(fig_dev, use_container_width=True)
    st.caption("🔴 Above diabetic mean  |  🟢 Below diabetic mean")


with tab3:
    st.markdown('<div class="section-title">Class Distribution in Training Data</div>', unsafe_allow_html=True)

    col_x, col_y = st.columns(2)

    with col_x:
        fig_pie = go.Figure(go.Pie(
            labels=["Non-Diabetic", "Diabetic"],
            values=[non_diabetic_count, diabetic_count],
            hole=0.55,
            marker=dict(
                colors=["rgba(34,197,94,0.7)","rgba(239,68,68,0.7)"],
                line=dict(color=["#4ade80","#f87171"], width=2),
            ),
            textinfo="label+percent",
            textfont=dict(color="#e2e8f0", size=13),
        ))
        fig_pie.update_layout(
            height=320, margin=dict(t=20,b=20,l=20,r=20),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            showlegend=False,
            annotations=[dict(text=f"{total_samples}<br><span style='font-size:11px'>Samples</span>",
                              showarrow=False, font=dict(size=20,color="#818cf8"))],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_y:
        # Feature means comparison horizontal bar
        fig_means = go.Figure()
        fig_means.add_trace(go.Bar(
            name="Non-Diabetic", y=features, x=model.theta_[0],
            orientation='h', marker_color="rgba(34,197,94,0.65)",
            marker_line=dict(color="#4ade80", width=1),
        ))
        fig_means.add_trace(go.Bar(
            name="Diabetic", y=features, x=model.theta_[1],
            orientation='h', marker_color="rgba(239,68,68,0.65)",
            marker_line=dict(color="#f87171", width=1),
        ))
        fig_means.update_layout(
            height=320, margin=dict(t=20,b=20,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", barmode="group",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
            title=dict(text="Feature Means by Class", font=dict(color="#94a3b8", size=13)),
        )
        st.plotly_chart(fig_means, use_container_width=True)

    # Feature variances
    st.markdown('<div class="section-title">Feature Variance (Gaussian Spread) per Class</div>', unsafe_allow_html=True)
    fig_var = make_subplots(rows=2, cols=4,
                            subplot_titles=features,
                            vertical_spacing=0.2,
                            horizontal_spacing=0.08)
    colors_cls  = ["#4ade80", "#f87171"]
    fcolors_cls = ["rgba(74,222,128,0.15)", "rgba(248,113,113,0.15)"]
    for idx, feat in enumerate(features):
        row = idx // 4 + 1
        col = idx % 4  + 1
        for cls in range(2):
            mu  = model.theta_[cls][idx]
            var = model.var_[cls][idx]
            std = np.sqrt(var)
            xs  = np.linspace(mu - 3*std, mu + 3*std, 100)
            ys  = (1/(std*np.sqrt(2*np.pi))) * np.exp(-0.5*((xs-mu)/std)**2)
            fig_var.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines",
                           line=dict(color=colors_cls[cls], width=2),
                           name=["Non-DM","DM"][cls],
                           showlegend=(idx == 0),
                           fill="tozeroy",
                           fillcolor=fcolors_cls[cls] ),
                row=row, col=col
            )
    fig_var.update_layout(
        height=420, margin=dict(t=40,b=20,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)"),
    )
    for ax in fig_var.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig_var.layout[ax].update(
                showgrid=False, zeroline=False,
                tickfont=dict(color="#64748b", size=9)
            )
    for ann in fig_var.layout.annotations:
        ann.update(font=dict(color="#94a3b8", size=11))
    st.plotly_chart(fig_var, use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(255,255,255,0.1);margin-top:2rem">
<p style="text-align:center;color:#475569;font-size:.8rem">
⚕️ <b>Medical Disclaimer</b>: This tool is for educational purposes only and does not constitute medical advice.<br>
Always consult a qualified healthcare professional for diagnosis and treatment.
</p>
""", unsafe_allow_html=True)