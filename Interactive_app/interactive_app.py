# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from scipy import stats
from scipy.special import inv_boxcox
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

st.set_page_config(
    page_title="Heteroskedasticity Playground",
    layout="centered",                 # narrower layout fits blogs
    initial_sidebar_state="collapsed", # hides sidebar by default
)

# Detect ?embed=true or ?embed=1 in the URL
try:
    q = st.query_params
    embed_flag = q.get("embed", "false")
    EMBED = (embed_flag in ("1", "true", "True", True))
except Exception:
    # Backward-compat for older Streamlit
    q = st.experimental_get_query_params()
    EMBED = str(q.get("embed", ["false"])[0]).lower() in ("1", "true")

# Compact styling in embed mode
if EMBED:
    st.markdown(
        """
        <style>
        /* Hide Streamlit chrome */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        /* Make content tighter & centered */
        .block-container {padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 900px;}
        /* Hide sidebar entirely (just in case) */
        [data-testid="stSidebar"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Optional: a compact height helper for charts (use when calling fig.update_layout)
DEFAULT_FIG_HEIGHT = 360 if EMBED else 520

# ------------------------
# Helpers
# ------------------------
def yeo_johnson_inverse(y, lmbda):
    # Inverse of Yeo–Johnson transform (scipy has forward only)
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    pos = y >= 0
    neg = ~pos
    # For y >= 0: y = ((x + 1)^λ - 1)/λ   if λ != 0;  y = log(x+1) if λ=0
    if lmbda != 0:
        out[pos] = np.power(y[pos] * lmbda + 1.0, 1.0 / lmbda) - 1.0
    else:
        out[pos] = np.exp(y[pos]) - 1.0
    # For y < 0:  y = -(((-x + 1)^(2-λ) - 1)/(2-λ)) if λ != 2; y = -log(-x + 1) if λ=2
    if lmbda != 2:
        out[neg] = 1.0 - np.power(1.0 - y[neg] * (2 - lmbda), 1.0 / (2.0 - lmbda))
        out[neg] *= -1.0
    else:
        out[neg] = 1.0 - np.exp(-y[neg])
        out[neg] *= -1.0
    return out

def make_weights(x, scheme, power=1.0, eps=1e-8):
    x = np.asarray(x, dtype=float)
    if scheme == "None (OLS)":
        return np.ones_like(x)
    if scheme == "1 / x":
        return 1.0 / np.clip(np.abs(x), eps, None)
    if scheme == "1 / x^2":
        return 1.0 / np.clip(np.abs(x) ** 2, eps, None)
    if scheme == "1 / exp(x)":
        return np.exp(-x)
    if scheme == "1 / |x|^p (choose p)":
        return 1.0 / np.clip(np.abs(x) ** power, eps, None)
    return np.ones_like(x)

def transform_y(y, how, lam, shift_for_pos):
    # Returns y_t, forward transform, plus an inverse function to go back
    y = np.asarray(y, dtype=float)
    if how == "Identity":
        return y, (lambda z: z), 0.0
    if how == "log(y)":
        shift = shift_for_pos - 1e-8  # ensure strictly >0
        return np.log(y + shift), (lambda z: np.exp(z) - shift), shift
    if how == "sqrt(y)":
        shift = shift_for_pos - 1e-8
        return np.sqrt(y + shift), (lambda z: np.maximum(z, 0.0) ** 2 - shift), shift
    if how == "Box-Cox (λ)":
        shift = shift_for_pos - 1e-8
        yt = stats.boxcox(y + shift, lmbda=lam)
        inv_fn = (lambda z: inv_boxcox(z, lam) - shift)
        return yt, inv_fn, shift
    if how == "Yeo-Johnson (λ)":
        # YJ does not require positivity; no shift needed
        yt = stats.yeojohnson(y, lmbda=lam)
        inv_fn = (lambda z: yeo_johnson_inverse(z, lam))
        return yt, inv_fn, 0.0
    return y, (lambda z: z), 0.0

def fit_simple(x, yt, weights, hc_kind=None):
    X = sm.add_constant(x)
    if np.allclose(weights, 1.0):
        model = sm.OLS(yt, X).fit(cov_type=("HC0" if hc_kind else "nonrobust"), cov_kwds=None if not hc_kind else {"use_correction": True})
    else:
        model = sm.WLS(yt, X, weights=weights).fit(cov_type=("HC0" if hc_kind else "nonrobust"))
    return model

def bp_white_tests(model, X):
    resid = model.resid
    y_fitted = model.fittedvalues
    bp = het_breuschpagan(resid, sm.add_constant(y_fitted))
    w = het_white(resid, X)  # White using full design
    # Return nicely
    bp_dict = {"Lagrange Multiplier": bp[0], "LM p-val": bp[1], "F-stat": bp[2], "F p-val": bp[3]}
    w_dict  = {"LM stat": w[0], "LM p-val": w[1], "F-stat": w[2], "F p-val": w[3]}
    return bp_dict, w_dict

# ------------------------
# Sidebar: data
# ------------------------
st.sidebar.header("Data")
data_source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Generate synthetic"], index=1)

if data_source == "Upload CSV":
    file = st.sidebar.file_uploader("Upload a CSV with at least two numeric columns", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
    else:
        st.info("Upload a CSV or switch to synthetic data.")
        st.stop()
    x_col = st.sidebar.selectbox("X column", options=df.columns)
    y_col = st.sidebar.selectbox("y column", options=df.columns)
    base = df[[x_col, y_col]].dropna().copy()
    base.columns = ["x", "y"]
else:
    st.sidebar.subheader("Synthetic data generator")
    n = st.sidebar.slider("n (rows)", 50, 2000, 300, step=50)
    beta0 = st.sidebar.number_input("β0 (intercept)", value=1.0)
    beta1 = st.sidebar.number_input("β1 (slope)", value=2.0)
    x_min, x_max = st.sidebar.slider("x range", -5.0, 5.0, (-1.0, 3.0))
    hetero_kind = st.sidebar.selectbox("Heteroskedastic noise", ["Constant (homosked.)", "σ ∝ |x|", "σ ∝ x^2 (scaled)", "σ ∝ exp(x)"])
    rng = np.random.default_rng(7)
    x = rng.uniform(x_min, x_max, size=n)
    mu = beta0 + beta1 * x
    if hetero_kind == "Constant (homosked.)":
        sigma = 1.0
    elif hetero_kind == "σ ∝ |x|":
        sigma = 0.5 + 0.6 * np.abs(x)
    elif hetero_kind == "σ ∝ x^2 (scaled)":
        sigma = 0.3 + 0.2 * (x - x.min())**2
    else:  # exp
        sigma = 0.2 + 0.15 * np.exp((x - x.mean()))
    y = mu + rng.normal(0, sigma)
    base = pd.DataFrame({"x": x, "y": y})

# ------------------------
# Sidebar: modeling choices
# ------------------------
st.sidebar.header("Modeling & Transforms")
resp_tf = st.sidebar.selectbox(
    "Response transform",
    ["Identity", "log(y)", "sqrt(y)", "Box-Cox (λ)", "Yeo-Johnson (λ)"],
    index=0
)
lam = st.sidebar.slider("λ (for Box-Cox/Yeo-Johnson)", -2.0, 3.0, 0.0, step=0.1)

w_scheme = st.sidebar.selectbox(
    "Weighted Least Squares (choose weights)",
    ["None (OLS)", "1 / x", "1 / x^2", "1 / exp(x)", "1 / |x|^p (choose p)"],
    index=0
)
power = st.sidebar.slider("p (if using 1/|x|^p)", 0.5, 4.0, 2.0, step=0.5)

hc = st.sidebar.selectbox("Robust (HC) SE", ["None", "HC0", "HC1", "HC2", "HC3"], index=0)
hc_kind = None if hc == "None" else hc

# ------------------------
# Prepare & transform
# ------------------------
data = base.copy()
data = data.replace([np.inf, -np.inf], np.nan).dropna()
x = data["x"].to_numpy()
y = data["y"].to_numpy()

# Shifts for positive-only transforms
shift_for_pos = max(1e-6, -(y.min()) + 1e-6) if y.min() <= 0 else 0.0
y_t, inv_transform, used_shift = transform_y(y, resp_tf, lam, shift_for_pos)

weights = make_weights(x, w_scheme, power=power)
X_design = sm.add_constant(x)

# Fit
model = fit_simple(x, y_t, weights, hc_kind=hc_kind)

# Predictions (sorted by x) and map back to original y-scale
order = np.argsort(x)
x_sorted = x[order]
X_sorted = sm.add_constant(x_sorted)
y_t_hat_sorted = model.predict(X_sorted)
y_hat_sorted = inv_transform(y_t_hat_sorted)

# Residuals (on transformed scale)
resid_t = model.resid
fitted_t = model.fittedvalues

# Diagnostics tests
bp, wt = bp_white_tests(model, X_design)

# R2 on transformed scale (from model) and an R2 on original scale (pseudo)
r2_t = model.rsquared
# Pseudo-R2 on original y-scale
ss_res = np.sum((y[order] - y_hat_sorted)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_orig = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

# ------------------------
# UI
# ------------------------
st.title("Interactive Simple Regression for Heteroskedastic Data")

with st.expander("What this app does"):
    st.markdown(
        """
- **Try transformations**: `log`, `sqrt`, `Box–Cox`, and `Yeo–Johnson` on the response to stabilize variance.
- **Try WLS**: choose weights like `1/x`, `1/x²`, `1/exp(x)`, or a generic `1/|x|^p`.
- **Robust SE**: toggle heteroskedasticity-consistent (HC) covariance for standard errors.
- **Diagnostics**: residuals vs fitted (on transformed scale) and Breusch–Pagan / White tests.
- **Plots** always show data and fitted line on the **original y-scale** for interpretability.
        """
    )

c1, c2 = st.columns([1.2, 1])

with c1:
    fig = px.scatter(
        data, x="x", y="y",
        labels={"x": "x", "y": "y (original scale)"},
        title="Data & Fitted Line (mapped back to original y-scale)",
    )
    fig.add_traces(
        px.line(x=x_sorted, y=y_hat_sorted, labels={"x": "x", "y": "ŷ (original scale)"}).data
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_resid = px.scatter(
        x=fitted_t, y=resid_t,
        labels={"x": "Fitted (transformed scale)", "y": "Residuals (transformed scale)"},
        title="Residuals vs Fitted (on transformed scale)",
    )
    st.plotly_chart(fig_resid, use_container_width=True)

with c2:
    st.subheader("Model summary (key stats)")
    st.markdown(f"**Response transform:** {resp_tf} (λ={lam:.1f})")
    st.markdown(f"**Weights:** {w_scheme}{' (p=' + str(power) + ')' if w_scheme=='1 / |x|^p (choose p)' else ''}")
    st.markdown(f"**Robust SE:** {hc if hc_kind else 'None'}")
    st.write("---")

    params = pd.DataFrame(
        {
            "coef": model.params,
            "std err": model.bse,
            "t": model.tvalues,
            "P>|t|": model.pvalues,
        }
    )
    st.dataframe(params.style.format(precision=4), use_container_width=True)

    metrics = pd.DataFrame(
        {
            "value": [
                model.aic,
                model.bic,
                r2_t,
                r2_orig,
            ]
        },
        index=["AIC (transformed)", "BIC (transformed)", "R² (transformed)", "Pseudo-R² (orig y)"],
    )
    st.table(metrics.style.format(precision=4))

    st.write("---")
    st.subheader("Heteroskedasticity tests")
    st.markdown("**Breusch–Pagan (on transformed scale residuals):**")
    st.write(pd.Series(bp).to_frame("value").style.format(precision=4))
    st.markdown("**White test (on transformed scale residuals):**")
    st.write(pd.Series(wt).to_frame("value").style.format(precision=4))

# Notes panel
with st.expander("Notes & Tips"):
    st.markdown(
        """
- **Interpreting transforms**:
  - `log(y)` often helps when variance grows multiplicatively with the mean.
  - `sqrt(y)` is common for count-like data.
  - `Box–Cox` requires strictly positive \(y\). The app auto-shifts \(y\) when needed and reverses the shift on predictions.
  - `Yeo–Johnson` works for zero/negative \(y\).
- **Weights**:
  - If residual spread increases with \(|x|\), try `1/x` or `1/x²` (avoid near \(x=0\)—we guard with a tiny epsilon).
  - If spread looks exponential in \(x\), try `1/exp(x)`.
  - Use the generic `1/|x|^p` slider to experiment.
- **Robust (HC) SE** change **standard errors** (and p-values) without changing the point estimates (for OLS).
- **Diagnostics**: Aim for a flat, equally-spread residuals-vs-fitted cloud; significant BP/White p-values suggest remaining heteroskedasticity.
        """
    )
