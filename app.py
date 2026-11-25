"""Underemployment Analytics dashboard. Run: python -m streamlit run app.py"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
import analysis as an

BASE = Path(__file__).parent
RAW_PATH = BASE / "raw_dataset_cleaned.csv"
ML_PATH = BASE / "ml_dataset.csv"

st.set_page_config(page_title="Underemployment Analytics", page_icon="📊", layout="wide")

BG, TEXT, MUTED, GRID = "#0E1117", "#E6EAF2", "#9AA4B5", "#2A3345"
BLUE, TEAL, AMBER, CORAL, VIOLET = "#5AA9E6", "#2DD4BF", "#F5B547", "#F2766B", "#A78BFA"
plt.rcParams.update({
    "figure.dpi": 115, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.linewidth": 1, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "legend.labelcolor": TEXT,
})

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1450px;}
h1 {font-weight: 750; letter-spacing: -0.6px;}
h2, h3 {font-weight: 680;}
[data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: 750;}
div[data-testid="stRadio"] > label {font-weight: 700;}
div[data-testid="stRadio"] div[role="radiogroup"] {gap: .45rem;}
div[data-testid="stRadio"] div[role="radiogroup"] label {background:#1b2433;padding:.55rem 1rem;border-radius:9px;border:1px solid #344158;}
.subtitle {color:#9AA4B5;font-size:1.02rem;margin-top:-.5rem;margin-bottom:1rem;}
.small-note {color:#9AA4B5;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_raw():
    return an.load_raw_cleaned(RAW_PATH)

@st.cache_data
def get_ml():
    return an.load_ml(ML_PATH)

@st.cache_resource
def get_models(_df):
    return an.train_models(_df)

@st.cache_data
def get_forecast(_df, model_name):
    return an.forecast(_df, model_name, years=5)

def show_df(frame, **kwargs):
    try:
        st.dataframe(frame, hide_index=True, use_container_width=True, **kwargs)
    except TypeError:
        st.dataframe(frame, hide_index=True, **kwargs)

def show_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def section_note(text):
    st.markdown(f'<p class="small-note">{text}</p>', unsafe_allow_html=True)

def sex_average(data):
    return pd.DataFrame({
        "Group": ["Both sexes", "Male", "Female"],
        "Average rate (%)": [data[c].mean() for c in an.SEX_TARGETS.values()],
    })

# Load data and show a useful error rather than a long traceback.
try:
    raw = get_raw()
    ml, median_ml = get_ml()
    configs = get_models(ml)
except Exception as exc:
    st.error(f"The app could not load or prepare the datasets: {exc}")
    st.info("Check that raw_dataset_cleaned.csv and ml_dataset.csv are beside app.py, and that their column names match the expected dataset columns.")
    st.stop()

# Top-level page navigation: dashboard, descriptive, forecasting, and separate model lab.
st.title("Underemployment Analytics")
st.markdown('<p class="subtitle">A data dashboard for exploring labor-force indicators, examining historical patterns, forecasting future trends, and evaluating supervised-learning models.</p>', unsafe_allow_html=True)
page = st.radio("Main navigation", ["Dashboard", "Descriptive Analytics", "Predictive Analytics", "Supervised Model"], horizontal=True, label_visibility="collapsed")
st.divider()

# Shared derived values
raw_yearly = raw.groupby("Year", as_index=False).agg(
    Underemployment=(an.TARGET, "mean"), LFPR=(an.LFPR, "mean"),
    Employment=("Employment Rate Both sexes", "mean"),
    Unemployment=("Unemployment Rate Both sexes", "mean"))
sex_avg = sex_average(raw)
model_accuracy = an.accuracy_of(configs)

if page == "Dashboard":
    st.header("Dashboard Overview")
    st.caption(f"Descriptive source: raw_dataset_cleaned.csv · Predictive source: ml_dataset.csv · Historical coverage: {int(raw['Year'].min())}–{int(raw['Year'].max())}")
    avg_under = raw[an.TARGET].mean()
    avg_lfpr = raw[an.LFPR].mean()
    latest = raw_yearly.iloc[-1]
    best_name = max(model_accuracy, key=model_accuracy.get)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average underemployment", f"{avg_under:.2f}%", help="Mean of the supplied descriptive dataset.")
    k2.metric("Average labor force participation", f"{avg_lfpr:.2f}%")
    k3.metric("Latest-year underemployment", f"{latest['Underemployment']:.2f}%", help=f"Average in {int(latest['Year'])}.")
    k4.metric("Top test accuracy", f"{model_accuracy[best_name]*100:.1f}%", help=f"{best_name}; holdout test set, not training accuracy.")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Historical underemployment trend")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(raw_yearly.Year, raw_yearly.Underemployment, marker="o", lw=2.5, color=BLUE)
        ax.fill_between(raw_yearly.Year, raw_yearly.Underemployment, alpha=.12, color=BLUE)
        ax.set_xlabel("Year"); ax.set_ylabel("Average underemployment rate (%)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=9))
        show_fig(fig)
    with right:
        st.subheader("Average rate by sex")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(sex_avg.Group, sex_avg["Average rate (%)"], color=[BLUE, TEAL, AMBER], width=.58)
        for bar, val in zip(bars, sex_avg["Average rate (%)"]):
            ax.text(bar.get_x()+bar.get_width()/2, val+.15, f"{val:.1f}%", ha="center", color=TEXT, fontweight="bold")
        ax.set_ylabel("Average underemployment rate (%)")
        ax.set_ylim(0, max(sex_avg["Average rate (%)"])*1.2)
        ax.grid(axis="x", visible=False)
        show_fig(fig)
    st.subheader("What each section contains")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Descriptive Analytics**\n\nHistorical summaries, distributions, yearly and monthly trends, comparisons by sex, and relationships among labor indicators.")
    c2.markdown("**Predictive Analytics**\n\nFive-year trend projection of the numeric underemployment rate, plus projected high-underemployment probability and downloadable results.")
    c3.markdown("**Supervised Model**\n\nNaive Bayes, Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine evaluation using a held-out test set.")
    section_note("Interpretation: dashboard indicators summarize the supplied datasets. Model accuracy measures classification performance on the test split; it does not mean that future labor-market values are known with certainty.")

elif page == "Descriptive Analytics":
    st.header("Descriptive Analytics")
    st.write("This section describes the historical data: its size, central tendency, variation, distribution, and changes over time. It does not train the supervised-learning models.")
    st.caption(f"Source: raw_dataset_cleaned.csv · {len(raw):,} records · {raw.shape[1]} columns · {int(raw.Year.min())}–{int(raw.Year.max())}")
    numeric = raw.select_dtypes(include=np.number).drop(columns=["Year", "Month"], errors="ignore")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Records", f"{len(raw):,}")
    q2.metric("Mean underemployment", f"{raw[an.TARGET].mean():.2f}%")
    q3.metric("Median underemployment", f"{raw[an.TARGET].median():.2f}%")
    q4.metric("Standard deviation", f"{raw[an.TARGET].std():.2f} pp")

    with st.expander("Dataset preview and descriptive statistics", expanded=True):
        st.markdown("**First 10 records**")
        show_df(raw.head(10))
        st.markdown("**Statistical summary**")
        stats = numeric.describe().T.rename(columns={"50%": "Median"}).round(3).reset_index(names="Indicator")
        show_df(stats)

    st.subheader("Historical patterns")
    a, b = st.columns(2)
    with a:
        st.markdown("**Average underemployment rate by year**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(raw_yearly.Year, raw_yearly.Underemployment, marker="o", color=BLUE, lw=2.3)
        ax.set_xlabel("Year"); ax.set_ylabel("Average rate (%)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))
        show_fig(fig)
    with b:
        st.markdown("**Distribution of underemployment rate**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(raw[an.TARGET], bins=14, kde=True, color=TEAL, edgecolor=BG, ax=ax)
        ax.axvline(raw[an.TARGET].mean(), color=AMBER, ls="--", label=f"Mean {raw[an.TARGET].mean():.2f}%")
        ax.axvline(raw[an.TARGET].median(), color=CORAL, ls=":", label=f"Median {raw[an.TARGET].median():.2f}%")
        ax.set_xlabel("Underemployment rate (%)"); ax.set_ylabel("Number of records"); ax.legend()
        show_fig(fig)

    st.subheader("Comparison by sex")
    st.write("The earlier pie chart was not an appropriate representation: the percentages are rates for different population groups, not parts of one whole. This chart compares the average rates directly instead.")
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(sex_avg.Group, sex_avg["Average rate (%)"], color=[BLUE, TEAL, AMBER], width=.55)
    for bar, val in zip(bars, sex_avg["Average rate (%)"]):
        ax.text(bar.get_x()+bar.get_width()/2, val+.12, f"{val:.2f}%", ha="center", color=TEXT, fontweight="bold")
    ax.set_ylabel("Average underemployment rate (%)"); ax.set_ylim(0, max(sex_avg["Average rate (%)"])*1.2)
    ax.grid(axis="x", visible=False)
    show_fig(fig)
    gap = raw[an.SEX_TARGETS["Male"]].mean() - raw[an.SEX_TARGETS["Female"]].mean()
    st.info(f"Across the supplied records, the average male underemployment rate is {abs(gap):.2f} percentage points {'higher' if gap >= 0 else 'lower'} than the female average. This is a descriptive difference, not an explanation of its cause.")

    c, d = st.columns(2)
    with c:
        st.markdown("**Average rate by month**")
        month_data = raw[raw.Month.between(1, 12)].groupby("Month")[an.TARGET].mean().reindex(range(1, 13))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([an.MONTH_NAMES[i] for i in month_data.index], month_data.values, color=VIOLET)
        ax.set_xlabel("Month"); ax.set_ylabel("Average rate (%)"); ax.tick_params(axis="x", rotation=35)
        show_fig(fig)
    with d:
        st.markdown("**Labor force participation vs underemployment**")
        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(raw[an.LFPR], raw[an.TARGET], c=raw.Year, cmap="viridis", s=36, alpha=.85)
        fig.colorbar(sc, ax=ax, label="Year")
        ax.set_xlabel("Labor force participation rate (%)"); ax.set_ylabel("Underemployment rate (%)")
        show_fig(fig)

    st.subheader("Key descriptive observations")
    low_year = raw_yearly.loc[raw_yearly.Underemployment.idxmin()]
    high_year = raw_yearly.loc[raw_yearly.Underemployment.idxmax()]
    st.markdown(f"The lowest yearly average in this file is **{low_year.Underemployment:.2f}% in {int(low_year.Year)}**, while the highest is **{high_year.Underemployment:.2f}% in {int(high_year.Year)}**. The overall mean is **{raw[an.TARGET].mean():.2f}%** and the median is **{raw[an.TARGET].median():.2f}%**.")
    st.markdown("These are descriptions of the records in the supplied file. They should not be interpreted as causal findings or as official estimates beyond the dataset's coverage.")

elif page == "Predictive Analytics":
    st.header("Predictive Analytics")
    st.write("This page focuses on the five-year outlook. It combines a numeric time-trend projection of the underemployment rate with a separate classification estimate of the probability of being above the ML dataset's median threshold.")
    f_name, f_acc = an.pick_forecast_model(configs)
    future, annual = get_forecast(ml, f_name)
    start_year, end_year = int(future.Year.min()), int(future.Year.max())
    st.caption(f"Forecast horizon: {start_year}–{end_year} · Classification model: {f_name} · Holdout accuracy: {f_acc*100:.2f}%")
    first_rate = float(future.groupby("Year").Projected_Underemployment_Rate.mean().iloc[0])
    last_rate = float(future.groupby("Year").Projected_Underemployment_Rate.mean().iloc[-1])
    avg_prob = float(future.Probability_High.mean())
    p1, p2, p3 = st.columns(3)
    p1.metric(f"Projected average rate · {start_year}", f"{first_rate:.2f}%")
    p2.metric(f"Projected average rate · {end_year}", f"{last_rate:.2f}%", delta=f"{last_rate-first_rate:+.2f} pp")
    p3.metric("Average high-category probability", f"{avg_prob*100:.1f}%")

    st.subheader("Five-year projected underemployment rate")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    hist = ml.groupby("Year")[an.TARGET].mean().reset_index()
    ax.plot(hist.Year, hist[an.TARGET], color=BLUE, marker="o", lw=2, label="Historical annual average")
    annual_rate = annual["Average_Projected_Underemployment"]
    ax.plot(annual.Year, annual_rate, color=AMBER, marker="o", ls="--", lw=2.5, label="Projected annual average")
    ax.axvline(int(ml.Year.max())+.5, color=MUTED, ls=":", label="Forecast begins")
    ax.set_xlabel("Year"); ax.set_ylabel("Underemployment rate (%)"); ax.legend()
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=12))
    show_fig(fig)

    l, r = st.columns(2)
    with l:
        st.markdown("**Projected high-underemployment probability by year**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(annual.Year, annual.Average_Probability_High*100, marker="o", color=CORAL, lw=2.2)
        ax.axhline(50, color=MUTED, ls="--", lw=1, label="50% reference")
        ax.set_xlabel("Year"); ax.set_ylabel("Average probability (%)"); ax.set_ylim(0, 100); ax.legend()
        show_fig(fig)
    with r:
        st.markdown("**Projected months classified as high risk**")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(annual.Year.astype(str), annual.High_Risk_Months, color=VIOLET)
        ax.set_xlabel("Year"); ax.set_ylabel("Number of months (0–12)"); ax.set_ylim(0, 12)
        show_fig(fig)

    st.subheader("Annual forecast table")
    annual_view = annual.rename(columns={"Average_Projected_Underemployment":"Average projected rate (%)", "Average_Probability_High":"Average probability of high category", "High_Risk_Months":"Months classified high"}).copy()
    annual_view["Average probability of high category"] = (annual_view["Average probability of high category"]*100).round(2)
    show_df(annual_view.round(3))
    st.download_button("Download five-year monthly forecast (CSV)", future.to_csv(index=False).encode("utf-8"), file_name=f"underemployment_forecast_{start_year}_{end_year}.csv", mime="text/csv")

    with st.expander("How to interpret this forecast and its limitations"):
        st.markdown("The numeric rate is projected by fitting a linear regression to Year and Month in the supplied ML dataset. The high/low probability is produced by the best-accuracy forecast-capable classifier, using projected labor-force participation inputs. The high/low threshold is the median underemployment rate in ml_dataset.csv. The two outputs answer different questions: one estimates a numeric rate trend, while the other estimates the chance of being above the historical median threshold.")
        st.warning("This is an exploratory, trend-based projection, not an official forecast. It assumes historical linear patterns continue, does not include economic shocks or policy changes, and does not provide prediction intervals. The 80/20 test accuracy is not a guarantee of future accuracy.")

elif page == "Supervised Model":
    st.header("Supervised Learning Model Evaluation")
    st.write("This is a separate model-evaluation page. Each algorithm learns to classify an observation as Low or High underemployment using the labor-force participation rates and month as input features.")
    st.caption(f"Source: ml_dataset.csv · {len(ml):,} rows after required-column cleaning · target threshold: median = {median_ml:.3f}% · stratified 80/20 train-test split")
    st.info(f"Class definition: Low = underemployment rate at or below {median_ml:.3f}%; High = above {median_ml:.3f}%. This median split creates a binary label for classification; it is not an official labor-market category.")

    selected = st.selectbox("Choose an algorithm to inspect", an.MODEL_NAMES, index=an.MODEL_NAMES.index(max(model_accuracy, key=model_accuracy.get)))
    model, xt, yt = configs[selected]
    metrics = an.weighted_metrics(model, xt, yt)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['Accuracy']*100:.2f}%")
    m2.metric("Weighted precision", f"{metrics['Precision']:.3f}")
    m3.metric("Weighted recall", f"{metrics['Recall']:.3f}")
    m4.metric("Weighted F1-score", f"{metrics['F1-score']:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = an.confusion_for(model, xt, yt)
        fig, ax = plt.subplots(figsize=(5.5, 4.4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
                    xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax,
                    annot_kws={"size":14,"weight":"bold"})
        ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
        show_fig(fig)
    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, auc = an.roc_for(model, xt, yt)
        fig, ax = plt.subplots(figsize=(5.5, 4.4))
        ax.plot(fpr, tpr, color=TEAL, lw=2.4, label=f"AUC = {auc:.3f}")
        ax.plot([0,1],[0,1],"--",color=MUTED,label="Random classifier")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend()
        show_fig(fig)

    st.markdown("**Classification report**")
    show_df(an.report_table(model, xt, yt))
    st.subheader("Compare all supervised algorithms")
    comparison = pd.DataFrame([{"Algorithm": name, "Accuracy (%)": model_accuracy[name]*100,
                                "Precision (weighted)": an.weighted_metrics(*configs[name])['Precision'],
                                "Recall (weighted)": an.weighted_metrics(*configs[name])['Recall'],
                                "F1-score (weighted)": an.weighted_metrics(*configs[name])['F1-score']}
                               for name in an.MODEL_NAMES]).sort_values("Accuracy (%)", ascending=False)
    show_df(comparison.round(4))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_data = comparison.sort_values("Accuracy (%)")
    bars = ax.barh(plot_data.Algorithm, plot_data["Accuracy (%)"], color=[TEAL if n == selected else BLUE for n in plot_data.Algorithm])
    for bar, value in zip(bars, plot_data["Accuracy (%)"]):
        ax.text(value+.5, bar.get_y()+bar.get_height()/2, f"{value:.1f}%", va="center", color=TEXT)
    ax.set_xlim(0, 110); ax.set_xlabel("Test accuracy (%)"); ax.grid(axis="y", visible=False)
    show_fig(fig)

    st.subheader("Test-set predictions")
    predictions = an.test_predictions(ml, model, xt, yt)
    show_df(predictions)
    st.download_button("Download test-set predictions (CSV)", predictions.to_csv(index=False).encode("utf-8"), file_name=f"{selected.lower().replace(' ','_')}_test_predictions.csv", mime="text/csv")
    with st.expander("Inputs, target, and evaluation notes"):
        st.markdown("**Input features:** labor force participation rate for both sexes, male, and female, plus Month. **Target:** binary category derived from the median underemployment rate in the ML dataset. **Evaluation:** stratified holdout split, with accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC reported on the test portion.")
        st.warning("A high score should be checked against data leakage, repeated or duplicated records, and how the ML dataset was constructed. The results describe this dataset and split; they do not prove that the model will perform equally well on new real-world data.")

st.divider()
st.caption("Underemployment Analytics · Descriptive analysis and exploratory predictive modeling")
