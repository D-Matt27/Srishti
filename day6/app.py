import io
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Oracle",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme  —  deep navy, aged gold, slate
# ─────────────────────────────────────────────────────────────────────────────
NAVY   = "#0B1D33"
GOLD   = "#C9A84C"
SLATE  = "#1E3A52"
CREAM  = "#F2ECD8"
RED    = "#C0392B"
GREEN  = "#1A8C5B"
LIGHT  = "#A8C4D8"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

  html, body, [class*="css"] {{
      font-family: 'Source Sans 3', sans-serif;
      background-color: {NAVY};
      color: {CREAM};
  }}
  .stApp {{ background-color: {NAVY}; }}

  [data-testid="stSidebar"] {{
      background-color: {SLATE} !important;
      border-right: 1px solid {GOLD}44;
  }}
  [data-testid="stSidebar"] * {{ color: {CREAM} !important; }}
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label {{
      color: {LIGHT} !important;
      font-weight: 600;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }}

  .oracle-title {{
      font-family: 'Playfair Display', serif;
      font-size: 3rem;
      font-weight: 900;
      color: {GOLD};
      text-align: center;
      letter-spacing: 0.04em;
      line-height: 1.1;
      text-shadow: 0 2px 20px #00000080;
  }}
  .oracle-sub {{
      text-align: center;
      font-size: 1rem;
      color: {LIGHT};
      letter-spacing: 0.2em;
      text-transform: uppercase;
      margin-top: -0.3rem;
      margin-bottom: 1.5rem;
  }}
  .divider {{
      border: none;
      border-top: 1px solid {GOLD}55;
      margin: 1.2rem 0;
  }}
  .prob-card {{
      background: linear-gradient(135deg, {SLATE}cc, {NAVY}cc);
      border: 1px solid {GOLD}66;
      border-radius: 16px;
      padding: 2rem 1.5rem;
      text-align: center;
  }}
  .prob-label {{
      font-family: 'Playfair Display', serif;
      font-size: 1rem;
      color: {LIGHT};
      text-transform: uppercase;
      letter-spacing: 0.18em;
      margin-bottom: 0.4rem;
  }}
  .prob-value {{
      font-family: 'Playfair Display', serif;
      font-size: 4.5rem;
      font-weight: 900;
      line-height: 1;
  }}
  .verdict {{
      font-size: 1.2rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.15em;
      margin-top: 0.8rem;
  }}
  .metric-row {{
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
      margin-top: 0.8rem;
  }}
  .metric-tile {{
      flex: 1;
      min-width: 100px;
      background: {SLATE}99;
      border: 1px solid {GOLD}33;
      border-radius: 10px;
      padding: 0.8rem;
      text-align: center;
  }}
  .metric-tile .val {{
      font-family: 'Playfair Display', serif;
      font-size: 1.6rem;
      font-weight: 700;
      color: {GOLD};
  }}
  .metric-tile .lbl {{
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: {LIGHT};
  }}
  .stTabs [data-baseweb="tab-list"] {{
      background: {SLATE}88;
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
      color: {LIGHT};
      font-weight: 600;
      font-size: 0.9rem;
      letter-spacing: 0.05em;
      border-radius: 7px;
      padding: 0.5rem 1.2rem;
  }}
  .stTabs [aria-selected="true"] {{
      background: {GOLD}22;
      color: {GOLD} !important;
  }}
  .section-hdr {{
      font-family: 'Playfair Display', serif;
      font-size: 1.4rem;
      color: {GOLD};
      border-bottom: 1px solid {GOLD}44;
      padding-bottom: 0.3rem;
      margin-top: 1.2rem;
      margin-bottom: 0.8rem;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib theme
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  NAVY,
    "axes.facecolor":    SLATE,
    "axes.edgecolor":    GOLD + "55",
    "axes.labelcolor":   CREAM,
    "xtick.color":       LIGHT,
    "ytick.color":       LIGHT,
    "text.color":        CREAM,
    "grid.color":        GOLD + "22",
    "grid.linestyle":    "--",
    "legend.facecolor":  SLATE,
    "legend.edgecolor":  GOLD + "44",
    "font.family":       "sans-serif",
})


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors train_model.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def build_input_row(pclass, sex, age, sibsp, parch, fare, embarked, title="Mr") -> pd.DataFrame:
    sex_enc      = 1 if sex.lower() == "female" else 0
    embarked_enc = {"S": 0, "C": 1, "Q": 2}.get(embarked, 0)
    family_size  = sibsp + parch + 1
    is_alone     = 1 if family_size == 1 else 0
    fare_pp      = fare / max(family_size, 1)

    if age <= 12:   age_group = 0
    elif age <= 18: age_group = 1
    elif age <= 35: age_group = 2
    elif age <= 60: age_group = 3
    else:           age_group = 4

    title_map = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4, "Unknown": 5}
    title_enc = title_map.get(title, 0)

    return pd.DataFrame([{
        "Pclass": pclass, "Sex_enc": sex_enc, "Age": age,
        "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "Embarked_enc": embarked_enc, "FamilySize": family_size,
        "IsAlone": is_alone, "FarePerPerson": fare_pp,
        "AgeGroup_enc": age_group, "Title_enc": title_enc,
    }])


# ─────────────────────────────────────────────────────────────────────────────
# Load pickle
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(pkl_bytes: bytes) -> dict | None:
    try:
        return pickle.load(io.BytesIO(pkl_bytes))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

@st.cache_resource
def load_model_from_path(path: str) -> dict | None:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# EDA plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def fig_survival_by(df, col, xlabel):
    fig, ax = plt.subplots(figsize=(6, 4))
    order = sorted(df[col].dropna().unique())
    sns.countplot(data=df, x=col, hue="Survived", palette={0: RED, 1: GREEN},
                  order=order, ax=ax, edgecolor=GOLD + "44", linewidth=0.6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Passengers", fontsize=10)
    ax.tick_params(axis="x")
    handles = [mpatches.Patch(color=RED, label="Died"),
               mpatches.Patch(color=GREEN, label="Survived")]
    ax.legend(handles=handles, framealpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_age_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    for survived, grp in df.groupby("Survived"):
        color = GREEN if survived == 1 else RED
        label = "Survived" if survived == 1 else "Died"
        sns.kdeplot(grp["Age"].dropna(), ax=ax, fill=True,
                    color=color, alpha=0.4, label=label, linewidth=2)
    ax.set_xlabel("Age")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.6)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fig_fare_boxplot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    df_plot = df[["Survived", "Fare"]].dropna()
    df_plot = df_plot.copy()
    df_plot["Status"] = df_plot["Survived"].map({0: "Died", 1: "Survived"})
    sns.boxplot(data=df_plot, x="Status", y="Fare",
                palette={"Died": RED, "Survived": GREEN}, ax=ax,
                linewidth=1.2,
                flierprops={"markerfacecolor": GOLD, "markersize": 3})
    ax.set_xlabel("")
    ax.set_ylabel("Fare (£)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_heatmap(df):
    num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
    corr = df[num_cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, linecolor=NAVY,
                annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
    plt.tight_layout()
    return fig


def fig_survival_rate_class_sex(df):
    grp = df.groupby(["Pclass", "Sex"])["Survived"].mean().reset_index()
    grp["Pclass"] = grp["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=grp, x="Pclass", y="Survived", hue="Sex",
                palette={"female": GOLD, "male": LIGHT}, ax=ax,
                edgecolor=NAVY, linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Survival Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def fig_feature_importance(importances: dict):
    items  = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace("_enc", "").replace("_", " ").title() for k, _ in items]
    values = [v for _, v in items]
    colors = [GOLD if v == max(values) else LIGHT for v in values]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                   edgecolor=NAVY, linewidth=0.5)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=CREAM)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="oracle-title">🚢 Titanic Survival Oracle</p>', unsafe_allow_html=True)
st.markdown('<p class="oracle-sub">Would you have survived the night of April 14, 1912?</p>',
            unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"<h2 style='color:{GOLD}; font-family:Playfair Display,serif;'>🔧 Setup</h2>",
                unsafe_allow_html=True)
    pkl_upload = st.file_uploader("Load model  (.pkl)", type=["pkl"])
    csv_upload = st.file_uploader("Load Titanic CSV  (for EDA)", type=["csv"])

    st.markdown(f"<hr style='border-color:{GOLD}44'>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{GOLD}; font-family:Playfair Display,serif;'>🧍 Passenger Profile</h3>",
                unsafe_allow_html=True)

    pclass   = st.selectbox("Ticket Class", [1, 2, 3],
                             format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
    sex      = st.selectbox("Sex", ["male", "female"])
    title    = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
    age      = st.slider("Age", 1, 80, 28)
    sibsp    = st.slider("Siblings / Spouses aboard", 0, 8, 0)
    parch    = st.slider("Parents / Children aboard", 0, 6, 0)
    fare     = st.slider("Fare paid (£)", 0, 512, 32)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                             format_func=lambda x: {"S": "Southampton (S)",
                                                     "C": "Cherbourg (C)",
                                                     "Q": "Queenstown (Q)"}[x])

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + CSV
# ─────────────────────────────────────────────────────────────────────────────

bundle = None
if pkl_upload:
    bundle = load_model(pkl_upload.read())
else:
    bundle = load_model_from_path("titanic_model.pkl")

df_titanic = None
if csv_upload:
    df_titanic = pd.read_csv(csv_upload)
else:
    try:
        df_titanic = pd.read_csv("titanic.csv")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_predict, tab_eda, tab_model = st.tabs([
    "⚡ Survival Predictor",
    "📊 Explore the Data",
    "🧠 Model Details",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    if bundle is None:
        st.warning("⚠️ No model loaded. Run `train_model.py` first, then upload `titanic_model.pkl` in the sidebar — or place it in the same folder as `app.py`.")
        st.code("python train_model.py --csv titanic.csv", language="bash")
        st.stop()

    pipeline = bundle["pipeline"]

    input_df = build_input_row(pclass, sex, age, sibsp, parch, fare, embarked, title)
    prob     = pipeline.predict_proba(input_df)[0][1]
    pct      = int(round(prob * 100))

    if pct >= 60:
        color, verdict, emoji = GREEN, "LIKELY SURVIVED", "✅"
    elif pct >= 40:
        color, verdict, emoji = GOLD, "IT'S UNCERTAIN", "⚠️"
    else:
        color, verdict, emoji = RED, "UNLIKELY TO SURVIVE", "💀"

    col_card, col_info = st.columns([1, 1], gap="large")

    with col_card:
        st.markdown(f"""
        <div class="prob-card">
            <div class="prob-label">Survival Probability</div>
            <div class="prob-value" style="color:{color};">{pct}%</div>
            <div class="verdict" style="color:{color};">{emoji} {verdict}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(prob)
        st.caption(f"Raw probability score: {prob:.4f}")

    with col_info:
        st.markdown('<p class="section-hdr">Passenger Summary</p>', unsafe_allow_html=True)
        family_size = sibsp + parch + 1
        fare_pp     = fare / max(family_size, 1)
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-tile"><div class="val">{['1st','2nd','3rd'][pclass-1]}</div><div class="lbl">Class</div></div>
          <div class="metric-tile"><div class="val">{sex.title()}</div><div class="lbl">Sex</div></div>
          <div class="metric-tile"><div class="val">{age}</div><div class="lbl">Age</div></div>
          <div class="metric-tile"><div class="val">{family_size}</div><div class="lbl">Family Size</div></div>
        </div>
        <div class="metric-row">
          <div class="metric-tile"><div class="val">£{fare}</div><div class="lbl">Fare Paid</div></div>
          <div class="metric-tile"><div class="val">£{fare_pp:.1f}</div><div class="lbl">Fare/Person</div></div>
          <div class="metric-tile"><div class="val">{embarked}</div><div class="lbl">Embarked</div></div>
          <div class="metric-tile"><div class="val">{title}</div><div class="lbl">Title</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-hdr" style="margin-top:1.5rem;">Historical Context</p>',
                    unsafe_allow_html=True)
        context_map = {
            (1,"female"): ("~97%","Women in 1st class had the highest survival rate."),
            (1,"male"):   ("~37%","Men in 1st class had a moderate survival rate."),
            (2,"female"): ("~92%","Women in 2nd class were well-prioritised for lifeboats."),
            (2,"male"):   ("~16%","2nd class men had poor survival odds."),
            (3,"female"): ("~50%","Women in 3rd class faced serious barriers to lifeboats."),
            (3,"male"):   ("~15%","3rd class men had the lowest survival rate of any group."),
        }
        hist_rate, hist_text = context_map.get((pclass, sex.lower()), ("~?",""))
        st.info(f"**Historical rate for {sex} / {pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'} class: {hist_rate}**\n\n{hist_text}")

    # Scenario comparison chart
    st.markdown('<p class="section-hdr" style="margin-top:2rem;">How class changes your odds</p>',
                unsafe_allow_html=True)
    class_probs = []
    for c in [1, 2, 3]:
        row = build_input_row(c, sex, age, sibsp, parch, fare, embarked, title)
        p   = pipeline.predict_proba(row)[0][1]
        class_probs.append((f"{c}{'st' if c==1 else 'nd' if c==2 else 'rd'} Class", p))

    fig_sc, ax_sc = plt.subplots(figsize=(7, 3))
    labels_sc = [x[0] for x in class_probs]
    vals_sc   = [x[1] for x in class_probs]
    colors_sc = [GREEN if v >= 0.6 else (GOLD if v >= 0.4 else RED) for v in vals_sc]
    bars = ax_sc.bar(labels_sc, vals_sc, color=colors_sc,
                     edgecolor=NAVY, linewidth=0.8, width=0.5)
    for bar, v in zip(bars, vals_sc):
        ax_sc.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.015,
                   f"{v:.0%}", ha="center", va="bottom",
                   color=CREAM, fontsize=11, fontweight="bold")
    ax_sc.set_ylim(0, 1.15)
    ax_sc.set_ylabel("Survival Probability")
    ax_sc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_sc.axhline(y=0.5, color=GOLD, linestyle="--", alpha=0.5, linewidth=0.8)
    ax_sc.set_title(f"{sex.title()}, Age {age}  —  survival odds by class",
                    color=GOLD, fontsize=11)
    ax_sc.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_sc)
    plt.close(fig_sc)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    if df_titanic is None:
        st.warning("⚠️ No CSV loaded. Upload `titanic.csv` in the sidebar (or place it in the same folder as `app.py`).")
    else:
        df = df_titanic.copy()
        total  = len(df)
        n_surv = int(df["Survived"].sum())
        surv_rt = n_surv / total

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Passengers", f"{total:,}")
        c2.metric("Survived", f"{n_surv:,}", f"{surv_rt:.1%}")
        c3.metric("Died", f"{total-n_surv:,}", f"-{1-surv_rt:.1%}")
        c4.metric("Missing Ages", f"{df['Age'].isna().sum():,}")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-hdr">Survival by Class</p>', unsafe_allow_html=True)
            f = fig_survival_by(df, "Pclass", "Ticket Class")
            st.pyplot(f); plt.close(f)
        with col2:
            st.markdown('<p class="section-hdr">Survival by Sex</p>', unsafe_allow_html=True)
            f = fig_survival_by(df, "Sex", "Sex")
            st.pyplot(f); plt.close(f)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown('<p class="section-hdr">Age Distribution</p>', unsafe_allow_html=True)
            f = fig_age_distribution(df)
            st.pyplot(f); plt.close(f)
        with col4:
            st.markdown('<p class="section-hdr">Fare by Survival</p>', unsafe_allow_html=True)
            f = fig_fare_boxplot(df)
            st.pyplot(f); plt.close(f)

        col5, col6 = st.columns(2)
        with col5:
            st.markdown('<p class="section-hdr">Survival Rate — Class × Sex</p>',
                        unsafe_allow_html=True)
            f = fig_survival_rate_class_sex(df)
            st.pyplot(f); plt.close(f)
        with col6:
            st.markdown('<p class="section-hdr">Correlation Matrix</p>', unsafe_allow_html=True)
            f = fig_heatmap(df)
            st.pyplot(f); plt.close(f)

        col7, _ = st.columns([1, 1])
        with col7:
            st.markdown('<p class="section-hdr">Survival by Port of Embarkation</p>',
                        unsafe_allow_html=True)
            f = fig_survival_by(df, "Embarked", "Port")
            st.pyplot(f); plt.close(f)

        with st.expander("📋 View raw data (first 200 rows)"):
            st.dataframe(df.head(200), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    if bundle is None:
        st.warning("⚠️ No model loaded.")
    else:
        metrics = bundle.get("metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",       f"{metrics.get('accuracy', 0):.2%}")
        c2.metric("ROC-AUC",        f"{metrics.get('roc_auc', 0):.4f}")
        c3.metric("5-Fold CV AUC",  f"{metrics.get('cv_auc', 0):.4f}")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<p class="section-hdr">Feature Importances</p>', unsafe_allow_html=True)
            if bundle.get("importances"):
                f = fig_feature_importance(bundle["importances"])
                st.pyplot(f); plt.close(f)

        with col_right:
            st.markdown('<p class="section-hdr">Model Architecture</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:{SLATE}99; border:1px solid {GOLD}33; border-radius:10px;
                        padding:1.2rem; font-size:0.9rem; line-height:1.9; font-family:monospace;">
            Pipeline:<br>
            &nbsp;&nbsp;1. <span style="color:{GOLD}">SimpleImputer</span> — strategy: median<br>
            &nbsp;&nbsp;2. <span style="color:{GOLD}">StandardScaler</span><br>
            &nbsp;&nbsp;3. <span style="color:{GOLD}">RandomForestClassifier</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;n_estimators : 300<br>
            &nbsp;&nbsp;&nbsp;&nbsp;max_depth    : 8<br>
            &nbsp;&nbsp;&nbsp;&nbsp;class_weight : balanced<br>
            &nbsp;&nbsp;&nbsp;&nbsp;random_state : 42<br><br>
            Train rows : {bundle.get('train_size','—')}<br>
            Test rows  : {bundle.get('test_size','—')}<br>
            Features   : {len(bundle.get('feature_names', []))}<br>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p class="section-hdr" style="margin-top:2rem;">How to reload the pickle</p>',
                    unsafe_allow_html=True)
        st.code("""
import pickle

# Load
with open("titanic_model.pkl", "rb") as f:
    bundle = pickle.load(f)

pipeline      = bundle["pipeline"]        # sklearn Pipeline
feature_names = bundle["feature_names"]   # list of feature names
importances   = bundle["importances"]     # {feature: importance}
metrics       = bundle["metrics"]         # accuracy, roc_auc, cv_auc

# Predict survival probability on new data
prob = pipeline.predict_proba(X_new)[0][1]
print(f"Survival probability: {prob:.2%}")
""", language="python")