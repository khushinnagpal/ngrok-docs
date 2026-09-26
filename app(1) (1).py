
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st

# ============================================================
# ShopEase FBDA 2026 — Streamlit Hero Dashboard
# ============================================================
GROUP_ID = "112_096_074"
SEED = 112096074
SAMPLE_SIZE = 2500
DATASET_NAME = "ShopEase Raw Orders Dataset"
KAGGLE_DATASET = "mernahabib/shopease-raw-orders-dataset"

random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(
    page_title="ShopEase | FBDA Dynamic Analytical Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helpers ----------
def norm(x):
    return (
        str(x).strip().lower()
        .replace(" ", "_").replace("-", "_").replace("/", "_")
        .replace("(", "").replace(")", "").replace("%", "pct")
    )

def find_col(df, aliases):
    nmap = {norm(c): c for c in df.columns}
    for a in aliases:
        if norm(a) in nmap:
            return nmap[norm(a)]
    for a in aliases:
        na = norm(a)
        for n, original in nmap.items():
            if na in n or n in na:
                return original
    return None

def clean_data(df):
    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]

    # Date conversion
    for c in x.columns:
        if any(k in norm(c) for k in ["date", "datetime", "timestamp", "time"]):
            converted = pd.to_datetime(x[c], errors="coerce")
            if converted.notna().mean() >= 0.50:
                x[c] = converted

    # Numeric-looking object conversion
    for c in x.columns:
        if x[c].dtype == "object":
            cleaned = (
                x[c].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[$₹€£]", "", regex=True)
                .str.replace("%", "", regex=False)
                .str.strip()
            )
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().mean() >= 0.85:
                x[c] = numeric
    return x

def semantic_columns(df):
    date_col = find_col(df, [
        "order_date", "date", "order_datetime", "purchase_date",
        "transaction_date", "created_at", "order_time", "datetime", "timestamp"
    ])
    amount_col = find_col(df, [
        "sales", "revenue", "amount", "total_amount", "order_amount",
        "total", "order_value", "price", "selling_price", "gross_revenue", "net_revenue"
    ])
    qty_col = find_col(df, ["quantity", "qty", "units", "quantity_ordered", "units_sold"])
    profit_col = find_col(df, ["profit", "net_profit", "gross_profit"])
    return date_col, amount_col, qty_col, profit_col

@st.cache_data(show_spinner=False)
def load_sample():
    candidates = [
        Path("data/shopease_sample_112_096_074.csv"),
        Path("shopease_sample_112_096_074.csv"),
        Path("data/shopease_sample.csv"),
    ]
    for p in candidates:
        if p.exists():
            return pd.read_csv(p), str(p)

    # Fallback for local execution if the raw Kaggle dataset is available.
    for p in [Path("."), Path("data")]:
        for csv in sorted(p.glob("*.csv")):
            if "shopease_sample" not in csv.name.lower():
                return pd.read_csv(csv), str(csv)

    raise FileNotFoundError(
        "Sample CSV not found. Run the Colab notebook first; it creates "
        "data/shopease_sample_112_096_074.csv."
    )

def ci_mean(x, confidence=0.95):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan, np.nan
    m = x.mean()
    h = stats.t.ppf((1 + confidence) / 2, len(x) - 1) * stats.sem(x)
    return m - h, m + h

# ---------- Load ----------
try:
    raw, source = load_sample()
    df = clean_data(raw)
except Exception as e:
    st.error(str(e))
    st.stop()

date_col, amount_col, qty_col, profit_col = semantic_columns(df)

# ---------- Sidebar ----------
st.sidebar.title("🛒 ShopEase Analytics")
st.sidebar.caption(f"Group: {GROUP_ID}  |  Fixed seed: {SEED}")
st.sidebar.markdown("### Interactive Controls")

cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = df.select_dtypes(include=np.number).columns.tolist()
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

default_cat = find_col(df, [
    "product_category", "category", "product_type", "segment",
    "payment_method", "payment_mode", "status", "state", "city"
])
default_num = amount_col if amount_col in num_cols else (num_cols[0] if num_cols else None)

selected_cat = None
selected_num = None

if cat_cols:
    selected_cat = st.sidebar.selectbox(
        "Categorical variable", cat_cols,
        index=cat_cols.index(default_cat) if default_cat in cat_cols else 0
    )

if num_cols:
    selected_num = st.sidebar.selectbox(
        "Numeric variable", num_cols,
        index=num_cols.index(default_num) if default_num in num_cols else 0
    )

filtered = df.copy()

if date_col and df[date_col].notna().any():
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[filtered[date_col].between(start, end)]

if selected_cat:
    options = sorted(filtered[selected_cat].fillna("Missing").astype(str).unique())
    chosen = st.sidebar.multiselect(
        f"Filter {selected_cat}", options, default=options
    )
    if chosen:
        filtered = filtered[
            filtered[selected_cat].fillna("Missing").astype(str).isin(chosen)
        ]

# ---------- Header ----------
st.title("🛒 ShopEase — Dynamic Analytical Dashboard")
st.markdown(
    f"**FBDA 2026 | {DATASET_NAME}**  \n"
    f"Fixed project sample: **{SAMPLE_SIZE:,} records** | "
    f"Current filtered view: **{len(filtered):,} records** | "
    f"Group: **{GROUP_ID}**"
)

# ---------- KPIs ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Records", f"{len(filtered):,}")

if amount_col:
    k2.metric("Total Sales / Amount", f"{pd.to_numeric(filtered[amount_col], errors='coerce').sum():,.2f}")
    k3.metric("Average Order Value", f"{pd.to_numeric(filtered[amount_col], errors='coerce').mean():,.2f}")
elif num_cols:
    k2.metric("Numeric Fields", f"{len(num_cols):,}")
    k3.metric("Numeric Mean", f"{filtered[num_cols[0]].mean():,.2f}")

if profit_col:
    k4.metric("Total Profit", f"{pd.to_numeric(filtered[profit_col], errors='coerce').sum():,.2f}")
elif qty_col:
    k4.metric("Total Quantity", f"{pd.to_numeric(filtered[qty_col], errors='coerce').sum():,.0f}")
else:
    k4.metric("Variables", f"{df.shape[1]:,}")

tabs = st.tabs([
    "Executive Overview",
    "Descriptive Statistics",
    "Categorical Analysis",
    "Visual Analytics",
    "Inferential Tests",
    "Regression",
    "Data Quality",
    "Raw Sample"
])

# ============================================================
# TAB 1 — Executive Overview
# ============================================================
with tabs[0]:
    st.subheader("Executive Overview")
    c1, c2 = st.columns(2)

    with c1:
        if date_col and amount_col:
            tmp = filtered[[date_col, amount_col]].dropna().copy()
            if not tmp.empty:
                tmp["Month"] = tmp[date_col].dt.to_period("M").astype(str)
                trend = tmp.groupby("Month")[amount_col].sum()
                fig, ax = plt.subplots(figsize=(8, 4))
                trend.plot(ax=ax, marker="o")
                ax.set_title("Monthly Sales / Amount Trend")
                ax.set_xlabel("Month")
                ax.set_ylabel("Total")
                ax.tick_params(axis="x", rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("No date + amount pair was detected for a time trend.")

    with c2:
        if selected_cat and selected_num:
            top = (
                filtered.groupby(selected_cat)[selected_num]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .sort_values()
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            top.plot(kind="barh", ax=ax)
            ax.set_title(f"Top Categories by {selected_num}")
            ax.set_xlabel("Total")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if selected_cat:
        freq = filtered[selected_cat].fillna("Missing").astype(str).value_counts()
        if not freq.empty:
            st.info(
                f"Highest-frequency category: **{freq.index[0]}** "
                f"({freq.iloc[0]:,} records). "
                f"Lowest-frequency category: **{freq.index[-1]}** "
                f"({freq.iloc[-1]:,} records)."
            )

# ============================================================
# TAB 2 — Descriptive Statistics
# ============================================================
with tabs[1]:
    st.subheader("Non-Categorical Data — Descriptive Statistics")
    if num_cols:
        desc = filtered[num_cols].describe(percentiles=[.25, .50, .75]).T
        desc["Range"] = desc["max"] - desc["min"]
        desc["Skewness"] = filtered[num_cols].skew()
        desc["Kurtosis"] = filtered[num_cols].kurtosis()
        st.dataframe(desc.round(4), use_container_width=True)

        central = pd.DataFrame({
            "Mean": filtered[num_cols].mean(),
            "Median": filtered[num_cols].median(),
            "Mode": [
                filtered[c].mode(dropna=True).iloc[0]
                if not filtered[c].mode(dropna=True).empty else np.nan
                for c in num_cols
            ]
        })
        st.subheader("Central Tendency")
        st.dataframe(central.round(4), use_container_width=True)

    if cat_cols:
        st.subheader("Categorical Frequency & Relative Frequency")
        cat_choice = st.selectbox("Select categorical field", cat_cols, key="freq_cat")
        freq = (
            filtered[cat_choice].fillna("Missing").astype(str)
            .value_counts()
            .rename_axis(cat_choice)
            .reset_index(name="Count")
        )
        freq["Relative Frequency"] = freq["Count"] / freq["Count"].sum()
        freq["Relative Frequency (%)"] = freq["Relative Frequency"] * 100
        st.dataframe(freq.round(4), use_container_width=True)

# ============================================================
# TAB 3 — Categorical Analysis
# ============================================================
with tabs[2]:
    st.subheader("Categorical Analysis")
    if selected_cat:
        freq = filtered[selected_cat].fillna("Missing").astype(str).value_counts()

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            freq.head(10).sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"Top 10 Frequencies — {selected_cat}")
            ax.set_xlabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            pie = freq.head(6)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.pie(pie.values, labels=pie.index, autopct="%1.1f%%")
            ax.set_title(f"Category Share — {selected_cat}")
            st.pyplot(fig)
            plt.close(fig)

        if selected_num:
            group_stats = (
                filtered.groupby(selected_cat)[selected_num]
                .agg(["count", "mean", "median", "std", "min", "max", "sum"])
                .sort_values("mean", ascending=False)
            )
            st.subheader(f"{selected_num} by {selected_cat}")
            st.dataframe(group_stats.round(4), use_container_width=True)

# ============================================================
# TAB 4 — Visual Analytics
# ============================================================
with tabs[3]:
    st.subheader("Required Visual Analytics")

    if len(num_cols) >= 2:
        x_col = st.selectbox("Scatter X", num_cols, index=0, key="scatter_x")
        y_col = st.selectbox("Scatter Y", num_cols, index=1, key="scatter_y")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=filtered, x=x_col, y=y_col, alpha=.65, ax=ax)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if selected_num:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(filtered[selected_num].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histogram — {selected_num}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.boxplot(x=filtered[selected_num], ax=ax)
            ax.set_title(f"Box-Whisker Plot — {selected_num}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        if selected_cat and filtered[selected_cat].nunique(dropna=True) <= 15:
            fig, ax = plt.subplots(figsize=(9, 5))
            sns.violinplot(
                data=filtered, x=selected_cat, y=selected_num,
                cut=0, inner="quartile", ax=ax
            )
            ax.set_title(f"Violin Plot: {selected_num} by {selected_cat}")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    if len(num_cols) >= 2:
        corr = filtered[num_cols].corr(method="pearson")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", center=0, ax=ax)
        ax.set_title("Pearson Correlation Heat Map")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        pair_cols = num_cols[:min(5, len(num_cols))]
        pair = filtered[pair_cols].dropna()
        if len(pair) > 500:
            pair = pair.sample(500, random_state=SEED)
        if len(pair) >= 2:
            st.markdown("**Pair Plot (maximum 5 numeric variables; capped at 500 observations)**")
            pg = sns.pairplot(pair)
            st.pyplot(pg.fig)
            plt.close(pg.fig)

# ============================================================
# TAB 5 — Inferential Tests
# ============================================================
with tabs[4]:
    st.subheader("Inferential Statistics")
    alpha = 0.05

    if selected_num:
        x = pd.to_numeric(filtered[selected_num], errors="coerce").dropna()
        if len(x) >= 2:
            lo, hi = ci_mean(x)
            st.write(f"**95% CI for mean of `{selected_num}`:** ({lo:.4f}, {hi:.4f})")

            st.markdown("### Normality Tests")
            if 3 <= len(x) <= 5000:
                sh = stats.shapiro(x)
                st.write(f"Shapiro–Wilk: statistic={sh.statistic:.4f}, p={sh.pvalue:.4g}")
            ks = stats.kstest(
                (x - x.mean()) / (x.std(ddof=1) if x.std(ddof=1) else 1),
                "norm"
            )
            st.write(f"Kolmogorov–Smirnov (standardized): statistic={ks.statistic:.4f}, p={ks.pvalue:.4g}")
            ad = stats.anderson(x, dist="norm")
            st.write(f"Anderson–Darling: statistic={ad.statistic:.4f}")
            jb = stats.jarque_bera(x)
            st.write(f"Jarque–Bera: statistic={jb.statistic:.4f}, p={jb.pvalue:.4g}")

            st.markdown("### One-Sample Mean Test")
            if np.isfinite(x).all():
                t0 = stats.ttest_1samp(x, 0)
                st.write(
                    f"One-sample t-test against mathematical benchmark 0: "
                    f"t={t0.statistic:.4f}, p={t0.pvalue:.4g}"
                )
                st.caption("The zero benchmark is statistical, not a business target.")

    if selected_cat and selected_num:
        st.markdown("### Mean / Variance / Non-Parametric Group Tests")
        tmp = filtered[[selected_cat, selected_num]].dropna()
        groups_named = [
            (name, g[selected_num].values)
            for name, g in tmp.groupby(selected_cat)
            if len(g) >= 2
        ]
        if len(groups_named) >= 2:
            names = [a for a, _ in groups_named]
            groups = [b for _, b in groups_named]

            if len(groups) == 2:
                tt = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                mw = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                v1, v2 = np.var(groups[0], ddof=1), np.var(groups[1], ddof=1)
                if v1 > 0 and v2 > 0:
                    F = v1 / v2
                    dfn, dfd = len(groups[0]) - 1, len(groups[1]) - 1
                    pF = 2 * min(stats.f.cdf(F, dfn, dfd), 1 - stats.f.cdf(F, dfn, dfd))
                    st.write(f"F-test of variance: F={F:.4f}, p={pF:.4g}")
                st.write(f"Welch t-test: t={tt.statistic:.4f}, p={tt.pvalue:.4g}")
                st.write(f"Mann–Whitney U: U={mw.statistic:.4f}, p={mw.pvalue:.4g}")
            else:
                an = stats.f_oneway(*groups)
                kw = stats.kruskal(*groups)
                st.write(f"One-way ANOVA: F={an.statistic:.4f}, p={an.pvalue:.4g}")
                st.write(f"Kruskal–Wallis: H={kw.statistic:.4f}, p={kw.pvalue:.4g}")

            lev = stats.levene(*groups, center="median")
            st.write(f"Levene: statistic={lev.statistic:.4f}, p={lev.pvalue:.4g}")

            # Bartlett is included because the rubric explicitly lists it.
            try:
                bart = stats.bartlett(*groups)
                st.write(f"Bartlett: statistic={bart.statistic:.4f}, p={bart.pvalue:.4g}")
            except Exception:
                pass

    if len(num_cols) >= 2:
        st.markdown("### Pearson & Spearman Correlation Tests")
        a = st.selectbox("Correlation X", num_cols, key="corr_x")
        b = st.selectbox("Correlation Y", num_cols, index=1 if len(num_cols) > 1 else 0, key="corr_y")
        if a != b:
            z = filtered[[a, b]].dropna()
            if len(z) >= 3:
                pear = stats.pearsonr(z[a], z[b])
                spear = stats.spearmanr(z[a], z[b])
                st.write(f"Pearson r={pear.statistic:.4f}, p={pear.pvalue:.4g}")
                st.write(f"Spearman ρ={spear.statistic:.4f}, p={spear.pvalue:.4g}")

    if selected_cat:
        st.markdown("### Chi-Square Tests")
        counts = filtered[selected_cat].fillna("Missing").astype(str).value_counts()
        if len(counts) >= 2:
            gof = stats.chisquare(counts.values)
            st.write(f"Chi-square goodness-of-fit: χ²={gof.statistic:.4f}, p={gof.pvalue:.4g}")

        low_cats = [c for c in cat_cols if 2 <= filtered[c].nunique(dropna=True) <= 10]
        if len(low_cats) >= 2:
            c2 = st.selectbox("Second categorical variable", low_cats, key="chi2_cat2")
            if c2 != selected_cat:
                ct = pd.crosstab(
                    filtered[selected_cat].fillna("Missing"),
                    filtered[c2].fillna("Missing")
                )
                if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                    chi2, p, dof, _ = stats.chi2_contingency(ct)
                    st.write(
                        f"Chi-square independence: χ²={chi2:.4f}, "
                        f"p={p:.4g}, df={dof}"
                    )
                    st.dataframe(ct, use_container_width=True)

    st.caption(
        "Wilcoxon signed-rank and Friedman tests are not automatically forced because "
        "they require paired/repeated-measures designs; this order dataset is normally "
        "independent transactional records. If the dataset contains a defensible paired "
        "structure, these tests should be added for that specific design."
    )

# ============================================================
# TAB 6 — Regression
# ============================================================
with tabs[5]:
    st.subheader("Regression / Predictive Analysis")

    if len(num_cols) >= 2:
        target = st.selectbox("Dependent variable (Y)", num_cols, key="reg_y")
        predictors = st.multiselect(
            "Independent variables (X)",
            [c for c in num_cols if c != target],
            default=[c for c in num_cols if c != target][:min(3, len(num_cols)-1)],
            key="reg_x"
        )
        if predictors:
            rd = filtered[[target] + predictors].dropna()
            if len(rd) >= max(30, len(predictors) * 10):
                X = sm.add_constant(rd[predictors])
                y = rd[target]
                model = sm.OLS(y, X).fit()
                st.text(model.summary().as_text())

                pred = model.predict(X)
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.scatterplot(x=y, y=pred, ax=ax)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("### Degree-2 Polynomial Regression")
        poly_x = st.selectbox(
            "Polynomial predictor", [c for c in num_cols if c != target],
            key="poly_x"
        )
        pdx = filtered[[target, poly_x]].dropna()
        if len(pdx) >= 30:
            X2 = sm.add_constant(pd.DataFrame({
                poly_x: pdx[poly_x],
                "x²": pdx[poly_x] ** 2
            }))
            pm = sm.OLS(pdx[target], X2).fit()
            st.text(pm.summary().as_text())

    if selected_cat and selected_num and filtered[selected_cat].nunique(dropna=True) <= 15:
        st.markdown("### Regression with Categorical Variable")
        rd = filtered[[selected_num, selected_cat]].dropna()
        if len(rd) >= 30:
            formula = f'Q("{selected_num}") ~ C(Q("{selected_cat}"))'
            try:
                cm = smf.ols(formula, data=rd).fit()
                st.text(cm.summary().as_text())
            except Exception as e:
                st.warning(f"Categorical regression unavailable: {e}")

    binary_cats = [c for c in cat_cols if filtered[c].dropna().nunique() == 2]
    if binary_cats and num_cols:
        st.markdown("### Logistic Regression")
        logit_target = st.selectbox("Binary categorical target", binary_cats, key="logit_target")
        logit_x = st.multiselect(
            "Logistic predictors", num_cols,
            default=num_cols[:min(3, len(num_cols))], key="logit_x"
        )
        if logit_x:
            ld = filtered[[logit_target] + logit_x].dropna().copy()
            classes = list(ld[logit_target].astype(str).unique())
            if len(classes) == 2 and len(ld) >= 50:
                ld["_y"] = (ld[logit_target].astype(str) == classes[1]).astype(int)
                try:
                    lm = sm.Logit(ld["_y"], sm.add_constant(ld[logit_x])).fit(disp=False)
                    st.text(lm.summary().as_text())
                    odds = pd.DataFrame({
                        "Coefficient": lm.params,
                        "Std Error": lm.bse,
                        "z": lm.tvalues,
                        "p-value": lm.pvalues,
                        "Odds Ratio": np.exp(lm.params)
                    })
                    st.dataframe(odds.round(4), use_container_width=True)
                except Exception as e:
                    st.warning(f"Logistic regression unavailable: {e}")

# ============================================================
# TAB 7 — Data Quality
# ============================================================
with tabs[6]:
    st.subheader("Data Quality Audit")
    quality = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Count": df.isna().sum(),
        "Missing %": df.isna().mean() * 100,
        "Unique Values": df.nunique(dropna=True),
    }).sort_values("Missing %", ascending=False)
    st.dataframe(quality.round(2), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Duplicate Rows", f"{df.duplicated().sum():,}")
    c4.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

# ============================================================
# TAB 8 — Raw Sample
# ============================================================
with tabs[7]:
    st.subheader("Fixed 2,500-Record Sample")
    st.dataframe(filtered, use_container_width=True, height=520)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Current Filtered CSV",
        data=csv,
        file_name=f"shopease_{GROUP_ID}_filtered.csv",
        mime="text/csv"
    )

st.caption(f"Source file: {source} | Fixed random seed: {SEED}")
