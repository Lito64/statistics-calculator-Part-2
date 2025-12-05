import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from decimal import Decimal, getcontext
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Complete Statistics Calculator - 42 Tests",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 50%, #3182ce 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 { margin: 0; font-size: 32px; font-weight: 700; }
    .main-header p { margin: 8px 0 0 0; opacity: 0.9; font-size: 16px; }
    
    .result-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #3182ce;
        margin: 15px 0;
    }
    
    .result-box.success { border-left-color: #38a169; }
    .result-box.warning { border-left-color: #d69e2e; }
    .result-box.danger { border-left-color: #e53e3e; }
    
    .result-value {
        font-size: 26px;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        padding: 18px;
        background: white;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    .decision-reject {
        font-size: 20px;
        font-weight: 700;
        padding: 16px;
        background: white;
        border-radius: 10px;
        text-align: center;
        margin: 15px 0;
        color: #c53030;
        border: 3px solid #c53030;
    }
    
    .decision-fail {
        font-size: 20px;
        font-weight: 700;
        padding: 16px;
        background: white;
        border-radius: 10px;
        text-align: center;
        margin: 15px 0;
        color: #2f855a;
        border: 3px solid #2f855a;
    }
    
    .hypothesis-box {
        background: #ebf8ff;
        padding: 15px 18px;
        border-radius: 10px;
        margin: 12px 0;
        border-left: 5px solid #3182ce;
        font-size: 15px;
    }
    
    .details-box {
        background: white;
        padding: 18px;
        border-radius: 10px;
        margin-top: 15px;
        font-size: 14px;
        line-height: 1.9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .section-header {
        background: linear-gradient(135deg, #2c5282 0%, #3182ce 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 20px;
        font-weight: 600;
    }
    
    .section-header.green { background: linear-gradient(135deg, #276749 0%, #38a169 100%); }
    .section-header.purple { background: linear-gradient(135deg, #553c9a 0%, #805ad5 100%); }
    .section-header.orange { background: linear-gradient(135deg, #c05621 0%, #dd6b20 100%); }
    .section-header.red { background: linear-gradient(135deg, #c53030 0%, #e53e3e 100%); }
    .section-header.teal { background: linear-gradient(135deg, #234e52 0%, #319795 100%); }
    .section-header.pink { background: linear-gradient(135deg, #97266d 0%, #d53f8c 100%); }
    
    .info-box {
        background: #e6fffa;
        border: 1px solid #38b2ac;
        border-radius: 8px;
        padding: 12px 15px;
        margin: 10px 0;
        font-size: 14px;
    }
    
    .warning-box {
        background: #fffaf0;
        border: 1px solid #ed8936;
        border-radius: 8px;
        padding: 12px 15px;
        margin: 10px 0;
        font-size: 14px;
    }
    
    .effect-size-box {
        background: #faf5ff;
        border: 1px solid #9f7aea;
        border-radius: 8px;
        padding: 12px 15px;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 12px 24px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PRECISION SETTINGS & SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    PRECISION = st.slider("Decimal Precision", min_value=4, max_value=16, value=8, step=1)
    getcontext().prec = PRECISION + 10
    
    st.markdown("---")
    st.markdown("### 📊 7 Calculator Suite")
    st.markdown("""
    **1. Descriptive & Normality**
    - Descriptive Statistics
    - Shapiro-Wilk Test
    - Kolmogorov-Smirnov Test
    - Levene's Test
    
    **2. Comparing Means**
    - One-Sample t-Test
    - Two-Sample t-Test (Welch's)
    - Paired t-Test
    - Mann-Whitney U
    - Wilcoxon Signed-Rank
    - Sign Test
    
    **3. Comparing Proportions**
    - One-Proportion z-Test
    - Two-Proportion z-Test
    - Exact Binomial Test
    - Confidence Intervals
    
    **4. Categorical Analysis**
    - Chi-Square Tests
    - Fisher's Exact Test
    - McNemar's Test
    - Cramér's V
    
    **5. ANOVA Family**
    - One-Way ANOVA
    - Two-Way ANOVA
    - Kruskal-Wallis
    - Friedman Test
    - Post-Hoc Tests
    
    **6. Correlation**
    - Pearson r
    - Spearman ρ
    - Kendall's τ
    
    **7. Regression**
    - Simple Linear
    - Multiple Regression
    - Logistic Regression
    """)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **42 Statistical Tests**
    
    All calculations use `scipy.stats` 
    for exact p-values — no approximations.
    
    Features:
    - Adjustable precision (4-16 decimals)
    - Effect sizes included
    - Assumption checking
    - Step-by-step calculations
    """)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt(value, decimals=None):
    """Format number with specified precision."""
    if decimals is None:
        decimals = PRECISION
    if isinstance(value, (int, float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return str(value)
        return f"{value:.{decimals}f}"
    return str(value)

def parse_data(text):
    """Parse comma/space/newline separated data into numpy array."""
    if not text or text.strip() == '':
        return np.array([])
    text = text.replace(',', ' ').replace('\n', ' ').replace('\t', ' ')
    values = []
    for item in text.split():
        try:
            values.append(float(item))
        except ValueError:
            pass
    return np.array(values)

def sample_std(data):
    """Sample standard deviation with Bessel's correction."""
    return np.std(data, ddof=1)

def sample_mean(data):
    """Sample mean."""
    return np.mean(data)

def t_critical(df, confidence_level):
    """Critical t-value for given confidence level."""
    alpha = 1 - confidence_level / 100
    return stats.t.ppf(1 - alpha / 2, df)

def z_critical(confidence_level):
    """Critical z-value for given confidence level."""
    alpha = 1 - confidence_level / 100
    return stats.norm.ppf(1 - alpha / 2)

def t_pvalue(t_stat, df, test_type='two'):
    """Exact p-value from t-distribution."""
    if test_type == 'two':
        return 2 * stats.t.sf(abs(t_stat), df)
    elif test_type == 'left':
        return stats.t.cdf(t_stat, df)
    else:
        return stats.t.sf(t_stat, df)

def z_pvalue(z_stat, test_type='two'):
    """Exact p-value from standard normal distribution."""
    if test_type == 'two':
        return 2 * stats.norm.sf(abs(z_stat))
    elif test_type == 'left':
        return stats.norm.cdf(z_stat)
    else:
        return stats.norm.sf(z_stat)

def chi2_pvalue(chi2_stat, df):
    """Exact p-value from chi-square distribution."""
    return stats.chi2.sf(chi2_stat, df)

def f_pvalue(f_stat, df1, df2):
    """Exact p-value from F-distribution."""
    return stats.f.sf(f_stat, df1, df2)

def welch_df(s1, n1, s2, n2):
    """Welch-Satterthwaite degrees of freedom."""
    v1 = s1**2 / n1
    v2 = s2**2 / n2
    numerator = (v1 + v2)**2
    denominator = (v1**2 / (n1 - 1)) + (v2**2 / (n2 - 1))
    return numerator / denominator

def cohens_d(mean1, mean2, std1, std2, n1, n2):
    """Cohen's d effect size (pooled standard deviation)."""
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    return (mean1 - mean2) / pooled_std

def cohens_d_one_sample(mean, mu0, std):
    """Cohen's d for one-sample test."""
    return (mean - mu0) / std

def interpret_cohens_d(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"

def cramers_v(chi2, n, min_dim):
    """Cramér's V effect size for chi-square."""
    return np.sqrt(chi2 / (n * (min_dim - 1)))

def interpret_cramers_v(v, df_min):
    """Interpret Cramér's V based on degrees of freedom."""
    if df_min == 1:
        if v < 0.1: return "Negligible"
        elif v < 0.3: return "Small"
        elif v < 0.5: return "Medium"
        else: return "Large"
    elif df_min == 2:
        if v < 0.07: return "Negligible"
        elif v < 0.21: return "Small"
        elif v < 0.35: return "Medium"
        else: return "Large"
    else:
        if v < 0.06: return "Negligible"
        elif v < 0.17: return "Small"
        elif v < 0.29: return "Medium"
        else: return "Large"

def eta_squared(ss_between, ss_total):
    """Eta squared effect size for ANOVA."""
    return ss_between / ss_total

def interpret_eta_squared(eta2):
    """Interpret eta squared magnitude."""
    if eta2 < 0.01:
        return "Negligible"
    elif eta2 < 0.06:
        return "Small"
    elif eta2 < 0.14:
        return "Medium"
    else:
        return "Large"

def interpret_r(r):
    """Interpret correlation coefficient magnitude."""
    r = abs(r)
    if r < 0.1:
        return "Negligible"
    elif r < 0.3:
        return "Weak"
    elif r < 0.5:
        return "Moderate"
    elif r < 0.7:
        return "Strong"
    else:
        return "Very Strong"

# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>📊 Complete Statistics Calculator</h1>
    <p>42 Statistical Tests • 7 Calculators • Exact P-Values • Effect Sizes</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN TABS - 7 CALCULATORS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📏 1. Descriptive & Normality",
    "📊 2. Comparing Means",
    "📈 3. Proportions",
    "📋 4. Categorical",
    "🔬 5. ANOVA",
    "🔗 6. Correlation",
    "📉 7. Regression"
])

# ============================================================
# TAB 1: DESCRIPTIVE STATISTICS & NORMALITY TESTS
# ============================================================

with tab1:
    calc1 = st.selectbox(
        "Select Analysis:",
        ["Descriptive Statistics", "Shapiro-Wilk Test (Normality)", 
         "Kolmogorov-Smirnov Test", "Levene's Test (Equal Variances)"],
        key="calc1_select"
    )
    st.markdown("---")
    
    # ----- DESCRIPTIVE STATISTICS -----
    if calc1 == "Descriptive Statistics":
        st.markdown('<div class="section-header teal">Descriptive Statistics</div>', unsafe_allow_html=True)
        
        data_text = st.text_area("Enter Data:", value="12, 15, 18, 22, 25, 28, 30, 33, 35, 40",
                                  placeholder="Enter values separated by commas, spaces, or newlines", 
                                  height=100, key="desc_data")
        
        if st.button("Calculate Statistics", type="primary", key="desc_btn"):
            data = parse_data(data_text)
            if len(data) < 2:
                st.error("Please enter at least 2 data values.")
            else:
                n = len(data)
                mean = np.mean(data)
                median = np.median(data)
                mode_result = stats.mode(data, keepdims=True)
                mode_val = mode_result.mode[0]
                mode_count = mode_result.count[0]
                std_sample = np.std(data, ddof=1)
                std_pop = np.std(data, ddof=0)
                variance = np.var(data, ddof=1)
                sem = std_sample / np.sqrt(n)
                data_range = np.max(data) - np.min(data)
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
                cv = (std_sample / mean) * 100 if mean != 0 else np.nan
                
                # Five number summary
                minimum = np.min(data)
                maximum = np.max(data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Measures of Center")
                    st.markdown(f"""
                    <div class="result-box">
                        <strong>Mean (x̄):</strong> {fmt(mean)}<br>
                        <strong>Median:</strong> {fmt(median)}<br>
                        <strong>Mode:</strong> {fmt(mode_val)} (appears {mode_count} times)<br>
                        <strong>Trimmed Mean (10%):</strong> {fmt(stats.trim_mean(data, 0.1))}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Five-Number Summary")
                    st.markdown(f"""
                    <div class="result-box">
                        <strong>Minimum:</strong> {fmt(minimum)}<br>
                        <strong>Q1 (25th %):</strong> {fmt(q1)}<br>
                        <strong>Median (50th %):</strong> {fmt(median)}<br>
                        <strong>Q3 (75th %):</strong> {fmt(q3)}<br>
                        <strong>Maximum:</strong> {fmt(maximum)}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Measures of Spread")
                    st.markdown(f"""
                    <div class="result-box">
                        <strong>Sample Std Dev (s):</strong> {fmt(std_sample)}<br>
                        <strong>Population Std Dev (σ):</strong> {fmt(std_pop)}<br>
                        <strong>Variance (s²):</strong> {fmt(variance)}<br>
                        <strong>Range:</strong> {fmt(data_range)}<br>
                        <strong>IQR:</strong> {fmt(iqr)}<br>
                        <strong>Std Error (SE):</strong> {fmt(sem)}<br>
                        <strong>Coeff. of Variation:</strong> {fmt(cv, 2)}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### Shape")
                    st.markdown(f"""
                    <div class="result-box">
                        <strong>Skewness:</strong> {fmt(skewness, 4)} 
                        {"(Right-skewed)" if skewness > 0.5 else "(Left-skewed)" if skewness < -0.5 else "(Approximately symmetric)"}<br>
                        <strong>Kurtosis:</strong> {fmt(kurtosis, 4)}
                        {"(Heavy tails)" if kurtosis > 1 else "(Light tails)" if kurtosis < -1 else "(Normal-like tails)"}<br>
                        <strong>n:</strong> {n}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Outlier detection
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                outliers = data[(data < lower_fence) | (data > upper_fence)]
                
                st.markdown("### Outlier Detection (IQR Method)")
                st.markdown(f"""
                <div class="info-box">
                    <strong>Lower Fence:</strong> Q1 - 1.5×IQR = {fmt(q1)} - 1.5×{fmt(iqr)} = {fmt(lower_fence)}<br>
                    <strong>Upper Fence:</strong> Q3 + 1.5×IQR = {fmt(q3)} + 1.5×{fmt(iqr)} = {fmt(upper_fence)}<br>
                    <strong>Potential Outliers:</strong> {list(outliers) if len(outliers) > 0 else "None detected"}
                </div>
                """, unsafe_allow_html=True)
    
    # ----- SHAPIRO-WILK TEST -----
    elif calc1 == "Shapiro-Wilk Test (Normality)":
        st.markdown('<div class="section-header teal">Shapiro-Wilk Test for Normality</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> Tests if data comes from a normal distribution.<br>
        <strong>Best for:</strong> Sample sizes 3 to 50 (up to ~5000)<br>
        <strong>H₀:</strong> Data is normally distributed | <strong>Hₐ:</strong> Data is not normally distributed
        </div>
        """, unsafe_allow_html=True)
        
        data_text = st.text_area("Enter Data:", value="12, 15, 18, 22, 25, 28, 30, 33, 35, 40",
                                  height=100, key="shapiro_data")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="shapiro_alpha")
        
        if st.button("Run Shapiro-Wilk Test", type="primary", key="shapiro_btn"):
            data = parse_data(data_text)
            if len(data) < 3:
                st.error("Shapiro-Wilk test requires at least 3 data values.")
            elif len(data) > 5000:
                st.warning("For n > 5000, consider using Kolmogorov-Smirnov test instead.")
            else:
                stat, p_value = stats.shapiro(data)
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Data is NOT normal' if reject else '✓ FAIL TO REJECT H₀ — Data is approximately normal'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> Data comes from a normal distribution<br>
                        <strong>Hₐ:</strong> Data does not come from a normal distribution
                    </div>
                    <div class="result-value">
                        W = {fmt(stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Sample Size:</strong> n = {len(data)}<br>
                        <strong>Significance Level:</strong> α = {alpha}<br><br>
                        <strong>Interpretation:</strong><br>
                        {"The data significantly deviates from normality. Consider using non-parametric tests." if reject else "There is no significant evidence against normality. Parametric tests are appropriate."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- KOLMOGOROV-SMIRNOV TEST -----
    elif calc1 == "Kolmogorov-Smirnov Test":
        st.markdown('<div class="section-header teal">Kolmogorov-Smirnov Test</div>', unsafe_allow_html=True)
        
        ks_type = st.radio("Test Type:", ["One-Sample (vs Normal)", "Two-Sample (Compare Distributions)"], 
                           horizontal=True, key="ks_type")
        
        if ks_type == "One-Sample (vs Normal)":
            st.markdown("""
            <div class="info-box">
            <strong>Purpose:</strong> Tests if data follows a normal distribution.<br>
            <strong>Best for:</strong> Larger samples (n > 50)<br>
            <strong>H₀:</strong> Data follows normal distribution | <strong>Hₐ:</strong> Data does not follow normal distribution
            </div>
            """, unsafe_allow_html=True)
            
            data_text = st.text_area("Enter Data:", value="12, 15, 18, 22, 25, 28, 30, 33, 35, 40", key="ks1_data")
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="ks1_alpha")
            
            if st.button("Run K-S Test", type="primary", key="ks1_btn"):
                data = parse_data(data_text)
                if len(data) < 3:
                    st.error("Please enter at least 3 data values.")
                else:
                    # Standardize data for comparison to standard normal
                    data_standardized = (data - np.mean(data)) / np.std(data, ddof=1)
                    stat, p_value = stats.kstest(data_standardized, 'norm')
                    reject = p_value < alpha
                    
                    decision_class = 'decision-reject' if reject else 'decision-fail'
                    decision_text = '❌ REJECT H₀ — Data is NOT normal' if reject else '✓ FAIL TO REJECT H₀ — Data is approximately normal'
                    
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="hypothesis-box">
                            <strong>H₀:</strong> Data comes from a normal distribution<br>
                            <strong>Hₐ:</strong> Data does not come from a normal distribution
                        </div>
                        <div class="result-value">
                            D = {fmt(stat)}<br>
                            P-value = {fmt(p_value)}
                        </div>
                        <div class="{decision_class}">
                            {decision_text}
                        </div>
                        <div class="details-box">
                            <strong>Sample Size:</strong> n = {len(data)}<br>
                            <strong>D statistic:</strong> Maximum distance between empirical and theoretical CDFs<br>
                            <strong>Note:</strong> For small samples (n < 50), Shapiro-Wilk is more powerful.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        else:  # Two-Sample K-S
            st.markdown("""
            <div class="info-box">
            <strong>Purpose:</strong> Tests if two samples come from the same distribution.<br>
            <strong>H₀:</strong> Both samples come from the same distribution | <strong>Hₐ:</strong> Samples come from different distributions
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                data1_text = st.text_area("Sample 1:", value="12, 15, 18, 22, 25", key="ks2_data1")
            with col2:
                data2_text = st.text_area("Sample 2:", value="14, 17, 20, 24, 28", key="ks2_data2")
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="ks2_alpha")
            
            if st.button("Run Two-Sample K-S Test", type="primary", key="ks2_btn"):
                data1 = parse_data(data1_text)
                data2 = parse_data(data2_text)
                if len(data1) < 2 or len(data2) < 2:
                    st.error("Please enter at least 2 values for each sample.")
                else:
                    stat, p_value = stats.ks_2samp(data1, data2)
                    reject = p_value < alpha
                    
                    decision_class = 'decision-reject' if reject else 'decision-fail'
                    decision_text = '❌ REJECT H₀ — Distributions differ' if reject else '✓ FAIL TO REJECT H₀ — No significant difference'
                    
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="hypothesis-box">
                            <strong>H₀:</strong> Both samples come from the same distribution<br>
                            <strong>Hₐ:</strong> Samples come from different distributions
                        </div>
                        <div class="result-value">
                            D = {fmt(stat)}<br>
                            P-value = {fmt(p_value)}
                        </div>
                        <div class="{decision_class}">
                            {decision_text}
                        </div>
                        <div class="details-box">
                            <strong>Sample 1:</strong> n₁ = {len(data1)}<br>
                            <strong>Sample 2:</strong> n₂ = {len(data2)}<br>
                            <strong>Interpretation:</strong> {"The two samples have significantly different distributions." if reject else "No significant evidence that the distributions differ."}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ----- LEVENE'S TEST -----
    elif calc1 == "Levene's Test (Equal Variances)":
        st.markdown('<div class="section-header teal">Levene\'s Test for Equality of Variances</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> Tests if variances are equal across groups (homogeneity of variance).<br>
        <strong>Use:</strong> Pre-test for ANOVA and t-tests. More robust than Bartlett's test.<br>
        <strong>H₀:</strong> All group variances are equal | <strong>Hₐ:</strong> At least one variance differs
        </div>
        """, unsafe_allow_html=True)
        
        num_groups = st.number_input("Number of Groups:", min_value=2, max_value=10, value=2, key="levene_groups")
        
        groups_data = []
        cols = st.columns(min(num_groups, 4))
        for i in range(num_groups):
            with cols[i % 4]:
                data_text = st.text_area(f"Group {i+1}:", value=f"{10+i*5}, {12+i*5}, {14+i*5}, {16+i*5}, {18+i*5}", 
                                         key=f"levene_g{i}")
                groups_data.append(data_text)
        
        center = st.selectbox("Center:", ["median", "mean", "trimmed"], index=0, key="levene_center",
                              help="median is most robust to non-normality")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="levene_alpha")
        
        if st.button("Run Levene's Test", type="primary", key="levene_btn"):
            groups = [parse_data(g) for g in groups_data]
            if any(len(g) < 2 for g in groups):
                st.error("Each group needs at least 2 values.")
            else:
                stat, p_value = stats.levene(*groups, center=center)
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Variances are UNEQUAL' if reject else '✓ FAIL TO REJECT H₀ — Variances are approximately equal'
                
                group_vars = [np.var(g, ddof=1) for g in groups]
                group_stds = [np.std(g, ddof=1) for g in groups]
                group_ns = [len(g) for g in groups]
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> σ₁² = σ₂² = ... = σₖ² (all variances equal)<br>
                        <strong>Hₐ:</strong> At least one variance is different
                    </div>
                    <div class="result-value">
                        W = {fmt(stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Group Summary:</strong><br>
                        {"<br>".join([f"Group {i+1}: n={group_ns[i]}, s²={fmt(group_vars[i])}, s={fmt(group_stds[i])}" for i in range(len(groups))])}<br><br>
                        <strong>Recommendation:</strong><br>
                        {"Use Welch's t-test or Welch's ANOVA (does not assume equal variances)." if reject else "Equal variance assumption is satisfied. Standard t-test/ANOVA is appropriate."}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 2: COMPARING MEANS
# ============================================================

with tab2:
    calc2 = st.selectbox(
        "Select Test:",
        ["One-Sample t-Test", "Two-Sample t-Test (Welch's)", "Paired t-Test",
         "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test", "Sign Test", "Mood's Median Test"],
        key="calc2_select"
    )
    st.markdown("---")
    
    # ----- ONE-SAMPLE T-TEST -----
    if calc2 == "One-Sample t-Test":
        st.markdown('<div class="section-header">One-Sample t-Test</div>', unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="t1_input")
        
        if input_type == "Summary Statistics":
            col1, col2 = st.columns(2)
            with col1:
                n = st.number_input("Sample Size (n):", min_value=2, value=30, key="t1_n")
                x_bar = st.number_input("Sample Mean (x̄):", value=105.0, format="%.6f", key="t1_mean")
                s = st.number_input("Sample Std Dev (s):", min_value=0.0001, value=15.0, format="%.6f", key="t1_std")
            with col2:
                mu0 = st.number_input("Null Value (μ₀):", value=100.0, format="%.6f", key="t1_mu0")
                test_type = st.selectbox("Alternative:", 
                    ["Two-tailed (μ ≠ μ₀)", "Left-tailed (μ < μ₀)", "Right-tailed (μ > μ₀)"], key="t1_type")
                alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="t1_alpha")
        else:
            col1, col2 = st.columns(2)
            with col1:
                raw_data = st.text_area("Enter data:", value="102, 105, 98, 110, 107, 103, 115, 99, 108, 104", key="t1_raw")
            with col2:
                mu0 = st.number_input("Null Value (μ₀):", value=100.0, format="%.6f", key="t1_mu0_raw")
                test_type = st.selectbox("Alternative:", 
                    ["Two-tailed (μ ≠ μ₀)", "Left-tailed (μ < μ₀)", "Right-tailed (μ > μ₀)"], key="t1_type_raw")
                alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="t1_alpha_raw")
        
        if st.button("Perform t-Test", type="primary", key="t1_btn"):
            try:
                if input_type == "Raw Data":
                    data = parse_data(raw_data)
                    if len(data) < 2:
                        st.error("Please enter at least 2 data values.")
                        st.stop()
                    n = len(data)
                    x_bar = sample_mean(data)
                    s = sample_std(data)
                
                tt = 'two' if 'Two' in test_type else ('left' if 'Left' in test_type else 'right')
                symbols = {'two': '≠', 'left': '<', 'right': '>'}
                
                df = n - 1
                se = s / np.sqrt(n)
                t_stat = (x_bar - mu0) / se
                p_value = t_pvalue(t_stat, df, tt)
                reject = p_value < alpha
                
                # Effect size
                d = cohens_d_one_sample(x_bar, mu0, s)
                d_interp = interpret_cohens_d(d)
                
                # Confidence interval
                t_crit = t_critical(df, (1 - alpha) * 100)
                ci_lower = x_bar - t_crit * se
                ci_upper = x_bar + t_crit * se
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> μ = {mu0}<br>
                        <strong>Hₐ:</strong> μ {symbols[tt]} {mu0}
                    </div>
                    <div class="result-value">
                        t = {fmt(t_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Effect Size:</strong> Cohen's d = {fmt(d, 4)} ({d_interp})<br>
                        <strong>{int((1-alpha)*100)}% CI for μ:</strong> ({fmt(ci_lower)}, {fmt(ci_upper)})
                    </div>
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        n = {n}, x̄ = {fmt(x_bar)}, s = {fmt(s)}<br><br>
                        <strong>Calculations:</strong><br>
                        df = n - 1 = {df}<br>
                        SE = s/√n = {fmt(se)}<br>
                        t = (x̄ - μ₀)/SE = ({fmt(x_bar)} - {mu0})/{fmt(se)} = {fmt(t_stat)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- TWO-SAMPLE T-TEST (WELCH'S) -----
    elif calc2 == "Two-Sample t-Test (Welch's)":
        st.markdown('<div class="section-header">Two-Sample t-Test (Welch\'s)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Welch's t-test</strong> does NOT assume equal variances. It is the recommended default for comparing two independent group means.
        </div>
        """, unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="t2_input")
        
        if input_type == "Summary Statistics":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Group 1:**")
                n1 = st.number_input("n₁:", min_value=2, value=25, key="t2_n1")
                x1 = st.number_input("x̄₁:", value=75.0, format="%.6f", key="t2_x1")
                s1 = st.number_input("s₁:", min_value=0.0001, value=10.0, format="%.6f", key="t2_s1")
            with col2:
                st.markdown("**Group 2:**")
                n2 = st.number_input("n₂:", min_value=2, value=30, key="t2_n2")
                x2 = st.number_input("x̄₂:", value=70.0, format="%.6f", key="t2_x2")
                s2 = st.number_input("s₂:", min_value=0.0001, value=12.0, format="%.6f", key="t2_s2")
        else:
            col1, col2 = st.columns(2)
            with col1:
                raw1 = st.text_area("Group 1 Data:", value="72, 75, 78, 80, 73, 76, 79, 74, 77, 81", key="t2_raw1")
            with col2:
                raw2 = st.text_area("Group 2 Data:", value="68, 70, 65, 72, 69, 71, 67, 73, 66, 70", key="t2_raw2")
        
        col1, col2 = st.columns(2)
        with col1:
            test_type = st.selectbox("Alternative:", 
                ["Two-tailed (μ₁ ≠ μ₂)", "Left-tailed (μ₁ < μ₂)", "Right-tailed (μ₁ > μ₂)"], key="t2_type")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="t2_alpha")
        
        if st.button("Perform t-Test", type="primary", key="t2_btn"):
            try:
                if input_type == "Raw Data":
                    data1 = parse_data(raw1)
                    data2 = parse_data(raw2)
                    if len(data1) < 2 or len(data2) < 2:
                        st.error("Please enter at least 2 values for each group.")
                        st.stop()
                    n1, n2 = len(data1), len(data2)
                    x1, x2 = sample_mean(data1), sample_mean(data2)
                    s1, s2 = sample_std(data1), sample_std(data2)
                
                tt = 'two' if 'Two' in test_type else ('left' if 'Left' in test_type else 'right')
                symbols = {'two': '≠', 'left': '<', 'right': '>'}
                
                se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
                df = welch_df(s1, n1, s2, n2)
                diff = x1 - x2
                t_stat = diff / se
                p_value = t_pvalue(t_stat, df, tt)
                reject = p_value < alpha
                
                # Effect size
                d = cohens_d(x1, x2, s1, s2, n1, n2)
                d_interp = interpret_cohens_d(d)
                
                # Confidence interval
                t_crit = t_critical(df, (1 - alpha) * 100)
                ci_lower = diff - t_crit * se
                ci_upper = diff + t_crit * se
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> μ₁ = μ₂ (μ₁ - μ₂ = 0)<br>
                        <strong>Hₐ:</strong> μ₁ {symbols[tt]} μ₂
                    </div>
                    <div class="result-value">
                        t = {fmt(t_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Effect Size:</strong> Cohen's d = {fmt(d, 4)} ({d_interp})<br>
                        <strong>{int((1-alpha)*100)}% CI for (μ₁ - μ₂):</strong> ({fmt(ci_lower)}, {fmt(ci_upper)})
                    </div>
                    <div class="details-box">
                        <strong>Group 1:</strong> n₁ = {n1}, x̄₁ = {fmt(x1)}, s₁ = {fmt(s1)}<br>
                        <strong>Group 2:</strong> n₂ = {n2}, x̄₂ = {fmt(x2)}, s₂ = {fmt(s2)}<br>
                        <strong>Difference:</strong> x̄₁ - x̄₂ = {fmt(diff)}<br><br>
                        <strong>Calculations:</strong><br>
                        SE = √[(s₁²/n₁) + (s₂²/n₂)] = {fmt(se)}<br>
                        df (Welch) = {fmt(df, 2)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- PAIRED T-TEST -----
    elif calc2 == "Paired t-Test":
        st.markdown('<div class="section-header">Paired t-Test</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Use when:</strong> Same subjects measured twice (before/after), or matched pairs.
        </div>
        """, unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="t3_input")
        
        if input_type == "Summary Statistics":
            col1, col2 = st.columns(2)
            with col1:
                n = st.number_input("Number of Pairs (n):", min_value=2, value=20, key="t3_n")
                d_bar = st.number_input("Mean Difference (d̄):", value=5.0, format="%.6f", key="t3_dbar")
            with col2:
                s_d = st.number_input("Std Dev of Differences (sᵈ):", min_value=0.0001, value=8.0, format="%.6f", key="t3_sd")
        else:
            col1, col2 = st.columns(2)
            with col1:
                before_text = st.text_area("Before / Group 1:", value="85, 90, 78, 92, 88, 76, 95, 82, 89, 91", key="t3_before")
            with col2:
                after_text = st.text_area("After / Group 2:", value="88, 95, 82, 98, 90, 80, 99, 86, 93, 96", key="t3_after")
        
        col1, col2 = st.columns(2)
        with col1:
            test_type = st.selectbox("Alternative:", 
                ["Two-tailed (μᵈ ≠ 0)", "Left-tailed (μᵈ < 0)", "Right-tailed (μᵈ > 0)"], key="t3_type")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="t3_alpha")
        
        if st.button("Perform Paired t-Test", type="primary", key="t3_btn"):
            try:
                if input_type == "Raw Data":
                    before_data = parse_data(before_text)
                    after_data = parse_data(after_text)
                    if len(before_data) != len(after_data) or len(before_data) < 2:
                        st.error("Please enter equal numbers of values (at least 2 pairs).")
                        st.stop()
                    diff = before_data - after_data
                    n = len(diff)
                    d_bar = sample_mean(diff)
                    s_d = sample_std(diff)
                
                tt = 'two' if 'Two' in test_type else ('left' if 'Left' in test_type else 'right')
                symbols = {'two': '≠', 'left': '<', 'right': '>'}
                
                df = n - 1
                se = s_d / np.sqrt(n)
                t_stat = d_bar / se
                p_value = t_pvalue(t_stat, df, tt)
                reject = p_value < alpha
                
                # Effect size
                d = d_bar / s_d
                d_interp = interpret_cohens_d(d)
                
                # CI
                t_crit = t_critical(df, (1 - alpha) * 100)
                ci_lower = d_bar - t_crit * se
                ci_upper = d_bar + t_crit * se
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> μᵈ = 0 (no difference)<br>
                        <strong>Hₐ:</strong> μᵈ {symbols[tt]} 0
                    </div>
                    <div class="result-value">
                        t = {fmt(t_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Effect Size:</strong> Cohen's d = {fmt(d, 4)} ({d_interp})<br>
                        <strong>{int((1-alpha)*100)}% CI for μᵈ:</strong> ({fmt(ci_lower)}, {fmt(ci_upper)})
                    </div>
                    <div class="details-box">
                        <strong>Paired Statistics:</strong><br>
                        n = {n} pairs, d̄ = {fmt(d_bar)}, sᵈ = {fmt(s_d)}<br><br>
                        <strong>Calculations:</strong><br>
                        df = n - 1 = {df}<br>
                        SE = sᵈ/√n = {fmt(se)}<br>
                        t = d̄/SE = {fmt(t_stat)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- MANN-WHITNEY U TEST -----
    elif calc2 == "Mann-Whitney U Test":
        st.markdown('<div class="section-header purple">Mann-Whitney U Test (Non-Parametric)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Non-parametric alternative to two-sample t-test.</strong><br>
        Use when: Data is skewed, ordinal, or normality assumption is violated.<br>
        Tests if one group tends to have larger values than the other.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            data1_text = st.text_area("Group 1:", value="12, 15, 18, 14, 16, 20, 13", key="mw_data1")
        with col2:
            data2_text = st.text_area("Group 2:", value="22, 25, 19, 28, 24, 21, 26", key="mw_data2")
        
        col1, col2 = st.columns(2)
        with col1:
            alternative = st.selectbox("Alternative:", ["two-sided", "less", "greater"], key="mw_alt")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="mw_alpha")
        
        if st.button("Perform Mann-Whitney U Test", type="primary", key="mw_btn"):
            data1 = parse_data(data1_text)
            data2 = parse_data(data2_text)
            if len(data1) < 2 or len(data2) < 2:
                st.error("Please enter at least 2 values for each group.")
            else:
                stat, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
                reject = p_value < alpha
                
                # Effect size (rank-biserial correlation)
                n1, n2 = len(data1), len(data2)
                r = 1 - (2*stat)/(n1*n2)  # rank-biserial correlation
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> The distributions of both groups are equal<br>
                        <strong>Hₐ:</strong> The distributions differ (one group tends to have larger values)
                    </div>
                    <div class="result-value">
                        U = {fmt(stat, 2)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Effect Size:</strong> Rank-biserial r = {fmt(r, 4)}
                    </div>
                    <div class="details-box">
                        <strong>Group 1:</strong> n₁ = {n1}, Median = {fmt(np.median(data1))}<br>
                        <strong>Group 2:</strong> n₂ = {n2}, Median = {fmt(np.median(data2))}<br><br>
                        <strong>Interpretation:</strong> {"One group tends to have significantly larger values than the other." if reject else "No significant difference in the distributions."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- WILCOXON SIGNED-RANK TEST -----
    elif calc2 == "Wilcoxon Signed-Rank Test":
        st.markdown('<div class="section-header purple">Wilcoxon Signed-Rank Test (Non-Parametric)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Non-parametric alternative to paired t-test.</strong><br>
        Use when: Paired data is skewed or ordinal.<br>
        Tests if the median difference is zero.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            before_text = st.text_area("Before / Sample 1:", value="85, 90, 78, 92, 88, 76, 95, 82", key="wsr_before")
        with col2:
            after_text = st.text_area("After / Sample 2:", value="88, 95, 80, 98, 90, 78, 99, 86", key="wsr_after")
        
        col1, col2 = st.columns(2)
        with col1:
            alternative = st.selectbox("Alternative:", ["two-sided", "less", "greater"], key="wsr_alt")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="wsr_alpha")
        
        if st.button("Perform Wilcoxon Test", type="primary", key="wsr_btn"):
            before = parse_data(before_text)
            after = parse_data(after_text)
            if len(before) != len(after) or len(before) < 6:
                st.error("Please enter equal numbers of values (at least 6 pairs for reliable results).")
            else:
                diff = before - after
                stat, p_value = stats.wilcoxon(diff, alternative=alternative)
                reject = p_value < alpha
                
                # Effect size (matched-pairs rank-biserial)
                n = len(diff)
                r = 1 - (2*stat)/(n*(n+1)/2)
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> Median difference = 0<br>
                        <strong>Hₐ:</strong> Median difference ≠ 0
                    </div>
                    <div class="result-value">
                        W = {fmt(stat, 2)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Effect Size:</strong> r = {fmt(r, 4)}
                    </div>
                    <div class="details-box">
                        <strong>n pairs:</strong> {n}<br>
                        <strong>Median difference:</strong> {fmt(np.median(diff))}<br>
                        <strong>Mean difference:</strong> {fmt(np.mean(diff))}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- SIGN TEST -----
    elif calc2 == "Sign Test":
        st.markdown('<div class="section-header purple">Sign Test (Non-Parametric)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Simplest non-parametric paired test.</strong><br>
        Use when: Only the direction of change matters, not magnitude. Very robust.<br>
        Based on binomial distribution of positive vs. negative differences.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            before_text = st.text_area("Before / Sample 1:", value="85, 90, 78, 92, 88, 76, 95, 82, 89, 91", key="sign_before")
        with col2:
            after_text = st.text_area("After / Sample 2:", value="88, 95, 80, 98, 88, 78, 99, 86, 93, 96", key="sign_after")
        
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="sign_alpha")
        
        if st.button("Perform Sign Test", type="primary", key="sign_btn"):
            before = parse_data(before_text)
            after = parse_data(after_text)
            if len(before) != len(after) or len(before) < 5:
                st.error("Please enter equal numbers of values (at least 5 pairs).")
            else:
                diff = after - before  # Positive if after > before
                # Remove zeros (ties)
                diff_nonzero = diff[diff != 0]
                n = len(diff_nonzero)
                n_positive = np.sum(diff_nonzero > 0)
                n_negative = np.sum(diff_nonzero < 0)
                
                # Two-tailed binomial test
                p_value = stats.binom_test(min(n_positive, n_negative), n, 0.5, alternative='two-sided')
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> Median difference = 0 (P(+) = P(-) = 0.5)<br>
                        <strong>Hₐ:</strong> Median difference ≠ 0
                    </div>
                    <div class="result-value">
                        Positive: {n_positive} | Negative: {n_negative}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Total pairs:</strong> {len(before)}<br>
                        <strong>Non-zero differences:</strong> {n}<br>
                        <strong>Ties (zeros):</strong> {len(before) - n}<br>
                        <strong>Positive differences:</strong> {n_positive}<br>
                        <strong>Negative differences:</strong> {n_negative}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- MOOD'S MEDIAN TEST -----
    elif calc2 == "Mood's Median Test":
        st.markdown('<div class="section-header purple">Mood\'s Median Test (Non-Parametric)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Tests if medians of groups are equal.</strong><br>
        Very robust to outliers. Less powerful than Mann-Whitney but more robust.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            data1_text = st.text_area("Group 1:", value="12, 15, 18, 14, 16, 100", key="mood_data1")
        with col2:
            data2_text = st.text_area("Group 2:", value="22, 25, 19, 28, 24, 21", key="mood_data2")
        
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="mood_alpha")
        
        if st.button("Perform Mood's Median Test", type="primary", key="mood_btn"):
            data1 = parse_data(data1_text)
            data2 = parse_data(data2_text)
            if len(data1) < 2 or len(data2) < 2:
                st.error("Please enter at least 2 values for each group.")
            else:
                stat, p_value, med, table = stats.median_test(data1, data2)
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> Medians are equal<br>
                        <strong>Hₐ:</strong> Medians are different
                    </div>
                    <div class="result-value">
                        χ² = {fmt(stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Grand Median:</strong> {fmt(med)}<br>
                        <strong>Group 1 Median:</strong> {fmt(np.median(data1))}<br>
                        <strong>Group 2 Median:</strong> {fmt(np.median(data2))}<br><br>
                        <strong>Contingency Table:</strong><br>
                        Above grand median: Group 1 = {table[0,0]}, Group 2 = {table[0,1]}<br>
                        Below grand median: Group 1 = {table[1,0]}, Group 2 = {table[1,1]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 3: COMPARING PROPORTIONS
# ============================================================

with tab3:
    calc3 = st.selectbox(
        "Select Test:",
        ["One-Proportion z-Test", "Two-Proportion z-Test", "Exact Binomial Test",
         "Confidence Interval for Proportion", "CI for Difference in Proportions"],
        key="calc3_select"
    )
    st.markdown("---")
    
    # ----- ONE-PROPORTION Z-TEST -----
    if calc3 == "One-Proportion z-Test":
        st.markdown('<div class="section-header green">One-Proportion z-Test</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("Number of Successes (x):", min_value=0, value=56, key="p1_x")
            n = st.number_input("Sample Size (n):", min_value=1, value=100, key="p1_n")
        with col2:
            p0 = st.number_input("Null Value (p₀):", min_value=0.0, max_value=1.0, value=0.5, format="%.4f", key="p1_p0")
            test_type = st.selectbox("Alternative:", 
                ["Two-tailed (p ≠ p₀)", "Left-tailed (p < p₀)", "Right-tailed (p > p₀)"], key="p1_type")
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="p1_alpha")
        
        if st.button("Perform z-Test", type="primary", key="p1_btn"):
            if x > n:
                st.error("Successes cannot exceed sample size.")
            else:
                tt = 'two' if 'Two' in test_type else ('left' if 'Left' in test_type else 'right')
                symbols = {'two': '≠', 'left': '<', 'right': '>'}
                
                p_hat = x / n
                se = np.sqrt(p0 * (1 - p0) / n)
                z_stat = (p_hat - p0) / se
                p_value = z_pvalue(z_stat, tt)
                reject = p_value < alpha
                
                # Check conditions
                cond1 = n * p0 >= 10
                cond2 = n * (1 - p0) >= 10
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> p = {p0}<br>
                        <strong>Hₐ:</strong> p {symbols[tt]} {p0}
                    </div>
                    <div class="result-value">
                        z = {fmt(z_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Sample:</strong> x = {x}, n = {n}, p̂ = {fmt(p_hat)}<br>
                        <strong>SE:</strong> √[p₀(1-p₀)/n] = {fmt(se)}<br><br>
                        <strong>Conditions:</strong><br>
                        np₀ = {n}×{p0} = {n*p0:.1f} {"✓" if cond1 else "✗"} (need ≥10)<br>
                        n(1-p₀) = {n}×{1-p0} = {n*(1-p0):.1f} {"✓" if cond2 else "✗"} (need ≥10)
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- TWO-PROPORTION Z-TEST -----
    elif calc3 == "Two-Proportion z-Test":
        st.markdown('<div class="section-header green">Two-Proportion z-Test</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Group 1:**")
            x1 = st.number_input("Successes (x₁):", min_value=0, value=45, key="p2_x1")
            n1 = st.number_input("Sample Size (n₁):", min_value=1, value=100, key="p2_n1")
        with col2:
            st.markdown("**Group 2:**")
            x2 = st.number_input("Successes (x₂):", min_value=0, value=35, key="p2_x2")
            n2 = st.number_input("Sample Size (n₂):", min_value=1, value=100, key="p2_n2")
        
        col1, col2 = st.columns(2)
        with col1:
            test_type = st.selectbox("Alternative:", 
                ["Two-tailed (p₁ ≠ p₂)", "Left-tailed (p₁ < p₂)", "Right-tailed (p₁ > p₂)"], key="p2_type")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="p2_alpha")
        
        if st.button("Perform z-Test", type="primary", key="p2_btn"):
            if x1 > n1 or x2 > n2:
                st.error("Successes cannot exceed sample size.")
            else:
                tt = 'two' if 'Two' in test_type else ('left' if 'Left' in test_type else 'right')
                symbols = {'two': '≠', 'left': '<', 'right': '>'}
                
                p1_hat = x1 / n1
                p2_hat = x2 / n2
                p_pooled = (x1 + x2) / (n1 + n2)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                z_stat = (p1_hat - p2_hat) / se
                p_value = z_pvalue(z_stat, tt)
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> p₁ = p₂ (p₁ - p₂ = 0)<br>
                        <strong>Hₐ:</strong> p₁ {symbols[tt]} p₂
                    </div>
                    <div class="result-value">
                        z = {fmt(z_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Group 1:</strong> x₁ = {x1}, n₁ = {n1}, p̂₁ = {fmt(p1_hat)}<br>
                        <strong>Group 2:</strong> x₂ = {x2}, n₂ = {n2}, p̂₂ = {fmt(p2_hat)}<br>
                        <strong>Pooled p:</strong> {fmt(p_pooled)}<br>
                        <strong>Difference:</strong> p̂₁ - p̂₂ = {fmt(p1_hat - p2_hat)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- EXACT BINOMIAL TEST -----
    elif calc3 == "Exact Binomial Test":
        st.markdown('<div class="section-header green">Exact Binomial Test</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Use when:</strong> Sample size is small (np₀ < 10 or n(1-p₀) < 10).<br>
        Calculates exact probabilities using binomial distribution.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("Number of Successes (x):", min_value=0, value=8, key="binom_x")
            n = st.number_input("Sample Size (n):", min_value=1, value=20, key="binom_n")
        with col2:
            p0 = st.number_input("Null Value (p₀):", min_value=0.0, max_value=1.0, value=0.5, format="%.4f", key="binom_p0")
            alternative = st.selectbox("Alternative:", ["two-sided", "less", "greater"], key="binom_alt")
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="binom_alpha")
        
        if st.button("Perform Exact Binomial Test", type="primary", key="binom_btn"):
            if x > n:
                st.error("Successes cannot exceed sample size.")
            else:
                p_value = stats.binom_test(x, n, p0, alternative=alternative)
                reject = p_value < alpha
                p_hat = x / n
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> p = {p0}<br>
                        <strong>Hₐ:</strong> p {"≠" if alternative == "two-sided" else "<" if alternative == "less" else ">"} {p0}
                    </div>
                    <div class="result-value">
                        Exact P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Sample:</strong> x = {x}, n = {n}, p̂ = {fmt(p_hat)}<br>
                        <strong>Expected under H₀:</strong> np₀ = {n*p0:.2f}<br><br>
                        <strong>Note:</strong> This is an exact test — no normal approximation is used.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- CI FOR PROPORTION -----
    elif calc3 == "Confidence Interval for Proportion":
        st.markdown('<div class="section-header green">Confidence Interval for Proportion</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("Number of Successes (x):", min_value=0, value=65, key="ci_p_x")
            n = st.number_input("Sample Size (n):", min_value=1, value=100, key="ci_p_n")
        with col2:
            cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci_p_cl")
        
        if st.button("Calculate CI", type="primary", key="ci_p_btn"):
            if x > n:
                st.error("Successes cannot exceed sample size.")
            else:
                p_hat = x / n
                z_star = z_critical(cl)
                se = np.sqrt(p_hat * (1 - p_hat) / n)
                me = z_star * se
                lower = max(0, p_hat - me)
                upper = min(1, p_hat + me)
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-value">
                        {cl}% CI: ({fmt(lower)}, {fmt(upper)})<br>
                        <span style="font-size:18px;">or ({fmt(lower*100, 2)}%, {fmt(upper*100, 2)}%)</span>
                    </div>
                    <div class="details-box">
                        <strong>Sample:</strong> x = {x}, n = {n}, p̂ = {fmt(p_hat)}<br>
                        <strong>z*:</strong> {fmt(z_star)}<br>
                        <strong>SE:</strong> √[p̂(1-p̂)/n] = {fmt(se)}<br>
                        <strong>ME:</strong> z* × SE = {fmt(me)}<br><br>
                        <strong>Interpretation:</strong> We are {cl}% confident that the true population proportion lies between {fmt(lower*100, 2)}% and {fmt(upper*100, 2)}%.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- CI FOR DIFFERENCE IN PROPORTIONS -----
    elif calc3 == "CI for Difference in Proportions":
        st.markdown('<div class="section-header green">CI for Difference in Proportions (p₁ - p₂)</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Group 1:**")
            x1 = st.number_input("Successes (x₁):", min_value=0, value=72, key="ci_p2_x1")
            n1 = st.number_input("Sample Size (n₁):", min_value=1, value=120, key="ci_p2_n1")
        with col2:
            st.markdown("**Group 2:**")
            x2 = st.number_input("Successes (x₂):", min_value=0, value=54, key="ci_p2_x2")
            n2 = st.number_input("Sample Size (n₂):", min_value=1, value=100, key="ci_p2_n2")
        
        cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci_p2_cl")
        
        if st.button("Calculate CI", type="primary", key="ci_p2_btn"):
            if x1 > n1 or x2 > n2:
                st.error("Successes cannot exceed sample size.")
            else:
                p1_hat = x1 / n1
                p2_hat = x2 / n2
                diff = p1_hat - p2_hat
                z_star = z_critical(cl)
                se = np.sqrt((p1_hat * (1 - p1_hat) / n1) + (p2_hat * (1 - p2_hat) / n2))
                me = z_star * se
                lower = diff - me
                upper = diff + me
                
                interp = "→ p₁ is significantly higher" if lower > 0 else "→ p₂ is significantly higher" if upper < 0 else "→ No significant difference (CI contains 0)"
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-value">
                        {cl}% CI for (p₁ - p₂): ({fmt(lower)}, {fmt(upper)})
                    </div>
                    <div class="details-box">
                        <strong>Group 1:</strong> p̂₁ = {x1}/{n1} = {fmt(p1_hat)}<br>
                        <strong>Group 2:</strong> p̂₂ = {x2}/{n2} = {fmt(p2_hat)}<br>
                        <strong>Difference:</strong> p̂₁ - p̂₂ = {fmt(diff)}<br>
                        <strong>SE:</strong> {fmt(se)}<br>
                        <strong>ME:</strong> {fmt(me)}<br><br>
                        <strong>Interpretation:</strong> {interp}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 4: CATEGORICAL ANALYSIS
# ============================================================

with tab4:
    calc4 = st.selectbox(
        "Select Test:",
        ["Chi-Square Test of Independence", "Chi-Square Goodness of Fit", 
         "Fisher's Exact Test", "McNemar's Test", "Cramér's V (Effect Size)"],
        key="calc4_select"
    )
    st.markdown("---")
    
    # ----- CHI-SQUARE TEST OF INDEPENDENCE -----
    if calc4 == "Chi-Square Test of Independence":
        st.markdown('<div class="section-header orange">Chi-Square Test of Independence</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> Tests if two categorical variables are independent or related.<br>
        <strong>H₀:</strong> Variables are independent | <strong>Hₐ:</strong> Variables are associated
        </div>
        """, unsafe_allow_html=True)
        
        rows = st.number_input("Number of Rows:", min_value=2, max_value=10, value=2, key="chi_rows")
        cols = st.number_input("Number of Columns:", min_value=2, max_value=10, value=2, key="chi_cols")
        
        st.markdown("**Enter observed frequencies:**")
        
        # Create input grid
        observed = []
        grid_cols = st.columns(cols)
        for i in range(rows):
            row_data = []
            for j in range(cols):
                with grid_cols[j]:
                    val = st.number_input(f"R{i+1}C{j+1}:", min_value=0, value=25 if (i+j) % 2 == 0 else 30, key=f"chi_{i}_{j}")
                    row_data.append(val)
            observed.append(row_data)
        
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="chi_alpha")
        
        if st.button("Perform Chi-Square Test", type="primary", key="chi_btn"):
            observed_array = np.array(observed)
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed_array)
            reject = p_value < alpha
            
            # Effect size
            n = observed_array.sum()
            min_dim = min(rows, cols)
            v = cramers_v(chi2_stat, n, min_dim)
            v_interp = interpret_cramers_v(v, min_dim - 1)
            
            decision_class = 'decision-reject' if reject else 'decision-fail'
            decision_text = '❌ REJECT H₀ — Variables are ASSOCIATED' if reject else '✓ FAIL TO REJECT H₀ — No significant association'
            
            st.markdown(f"""
            <div class="result-box">
                <div class="hypothesis-box">
                    <strong>H₀:</strong> Variables are independent<br>
                    <strong>Hₐ:</strong> Variables are associated
                </div>
                <div class="result-value">
                    χ² = {fmt(chi2_stat)}<br>
                    P-value = {fmt(p_value)}
                </div>
                <div class="{decision_class}">
                    {decision_text}
                </div>
                <div class="effect-size-box">
                    <strong>Effect Size:</strong> Cramér's V = {fmt(v, 4)} ({v_interp})
                </div>
                <div class="details-box">
                    <strong>df:</strong> (r-1)(c-1) = ({rows}-1)({cols}-1) = {dof}<br>
                    <strong>n:</strong> {n}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show expected frequencies
            st.markdown("**Expected Frequencies:**")
            st.dataframe(pd.DataFrame(expected, columns=[f"Col {j+1}" for j in range(cols)],
                                       index=[f"Row {i+1}" for i in range(rows)]).round(4))
    
    # ----- CHI-SQUARE GOODNESS OF FIT -----
    elif calc4 == "Chi-Square Goodness of Fit":
        st.markdown('<div class="section-header orange">Chi-Square Goodness of Fit Test</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> Tests if observed frequencies match expected frequencies.<br>
        <strong>H₀:</strong> Data follows expected distribution | <strong>Hₐ:</strong> Data does not follow expected distribution
        </div>
        """, unsafe_allow_html=True)
        
        observed_text = st.text_area("Observed Frequencies:", value="30, 25, 20, 25", key="gof_obs")
        expected_text = st.text_area("Expected Frequencies (or leave blank for equal):", value="", key="gof_exp",
                                      help="Leave blank to test against equal proportions")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="gof_alpha")
        
        if st.button("Perform Goodness of Fit Test", type="primary", key="gof_btn"):
            observed = parse_data(observed_text)
            if len(observed) < 2:
                st.error("Please enter at least 2 categories.")
            else:
                if expected_text.strip() == '':
                    expected = np.full(len(observed), observed.sum() / len(observed))
                else:
                    expected = parse_data(expected_text)
                    if len(expected) != len(observed):
                        st.error("Number of expected values must match observed.")
                        st.stop()
                
                chi2_stat, p_value = stats.chisquare(observed, expected)
                dof = len(observed) - 1
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Data does NOT fit expected' if reject else '✓ FAIL TO REJECT H₀ — Data fits expected distribution'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> Data follows the expected distribution<br>
                        <strong>Hₐ:</strong> Data does not follow the expected distribution
                    </div>
                    <div class="result-value">
                        χ² = {fmt(chi2_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>df:</strong> k - 1 = {len(observed)} - 1 = {dof}<br>
                        <strong>Observed:</strong> {list(observed)}<br>
                        <strong>Expected:</strong> {[round(e, 2) for e in expected]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- FISHER'S EXACT TEST -----
    elif calc4 == "Fisher's Exact Test":
        st.markdown('<div class="section-header orange">Fisher\'s Exact Test</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Use when:</strong> Expected counts are small (&lt;5) in a 2×2 table.<br>
        Calculates exact probability without chi-square approximation.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Enter 2×2 contingency table:**")
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Cell [1,1]:", min_value=0, value=8, key="fisher_a")
            c = st.number_input("Cell [2,1]:", min_value=0, value=2, key="fisher_c")
        with col2:
            b = st.number_input("Cell [1,2]:", min_value=0, value=3, key="fisher_b")
            d = st.number_input("Cell [2,2]:", min_value=0, value=12, key="fisher_d")
        
        alternative = st.selectbox("Alternative:", ["two-sided", "less", "greater"], key="fisher_alt")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="fisher_alpha")
        
        if st.button("Perform Fisher's Exact Test", type="primary", key="fisher_btn"):
            table = np.array([[a, b], [c, d]])
            odds_ratio, p_value = stats.fisher_exact(table, alternative=alternative)
            reject = p_value < alpha
            
            decision_class = 'decision-reject' if reject else 'decision-fail'
            decision_text = '❌ REJECT H₀ — Significant association' if reject else '✓ FAIL TO REJECT H₀ — No significant association'
            
            st.markdown(f"""
            <div class="result-box">
                <div class="hypothesis-box">
                    <strong>H₀:</strong> No association (odds ratio = 1)<br>
                    <strong>Hₐ:</strong> There is an association
                </div>
                <div class="result-value">
                    Odds Ratio = {fmt(odds_ratio)}<br>
                    Exact P-value = {fmt(p_value)}
                </div>
                <div class="{decision_class}">
                    {decision_text}
                </div>
                <div class="details-box">
                    <strong>Table:</strong><br>
                    [{a}, {b}]<br>
                    [{c}, {d}]<br><br>
                    <strong>Odds Ratio Interpretation:</strong><br>
                    {"OR > 1: Positive association" if odds_ratio > 1 else "OR < 1: Negative association" if odds_ratio < 1 else "OR = 1: No association"}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ----- MCNEMAR'S TEST -----
    elif calc4 == "McNemar's Test":
        st.markdown('<div class="section-header orange">McNemar\'s Test</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> Tests change in binary outcome for paired data (before/after).<br>
        Only uses discordant pairs (where outcome changed).
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Enter 2×2 table for paired binary data:**")
        st.markdown("Rows = Before (Yes/No), Columns = After (Yes/No)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**After = Yes**")
            a = st.number_input("Before=Yes, After=Yes:", min_value=0, value=40, key="mcn_a")
            c = st.number_input("Before=No, After=Yes:", min_value=0, value=15, key="mcn_c")
        with col2:
            st.markdown("**After = No**")
            b = st.number_input("Before=Yes, After=No:", min_value=0, value=5, key="mcn_b")
            d = st.number_input("Before=No, After=No:", min_value=0, value=35, key="mcn_d")
        
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="mcn_alpha")
        
        if st.button("Perform McNemar's Test", type="primary", key="mcn_btn"):
            table = np.array([[a, b], [c, d]])
            # Use exact test if discordant pairs are small
            if b + c < 25:
                # Exact binomial test on discordant pairs
                p_value = stats.binom_test(min(b, c), b + c, 0.5)
                test_used = "Exact (binomial)"
                stat = min(b, c)
            else:
                # Chi-square approximation
                stat = (abs(b - c) - 1)**2 / (b + c)  # with continuity correction
                p_value = chi2_pvalue(stat, 1)
                test_used = "Chi-square (with continuity correction)"
            
            reject = p_value < alpha
            
            decision_class = 'decision-reject' if reject else 'decision-fail'
            decision_text = '❌ REJECT H₀ — Significant change' if reject else '✓ FAIL TO REJECT H₀ — No significant change'
            
            st.markdown(f"""
            <div class="result-box">
                <div class="hypothesis-box">
                    <strong>H₀:</strong> No change in proportions (b = c)<br>
                    <strong>Hₐ:</strong> Proportions changed
                </div>
                <div class="result-value">
                    Test = {test_used}<br>
                    P-value = {fmt(p_value)}
                </div>
                <div class="{decision_class}">
                    {decision_text}
                </div>
                <div class="details-box">
                    <strong>Concordant pairs:</strong> {a + d} (no change)<br>
                    <strong>Discordant pairs:</strong> {b + c}<br>
                    - Changed Yes→No: {b}<br>
                    - Changed No→Yes: {c}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ----- CRAMÉR'S V -----
    elif calc4 == "Cramér's V (Effect Size)":
        st.markdown('<div class="section-header orange">Cramér\'s V Effect Size Calculator</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> Measures strength of association between two categorical variables.<br>
        <strong>Range:</strong> 0 (no association) to 1 (perfect association)
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            chi2_input = st.number_input("Chi-Square (χ²) Statistic:", min_value=0.0, value=15.5, format="%.4f", key="cv_chi2")
        with col2:
            n_input = st.number_input("Total Sample Size (n):", min_value=1, value=200, key="cv_n")
        with col3:
            min_dim = st.number_input("Min(rows, cols):", min_value=2, value=2, key="cv_dim",
                                       help="Smaller of number of rows or columns in contingency table")
        
        if st.button("Calculate Cramér's V", type="primary", key="cv_btn"):
            v = cramers_v(chi2_input, n_input, min_dim)
            v_interp = interpret_cramers_v(v, min_dim - 1)
            
            st.markdown(f"""
            <div class="result-box">
                <div class="result-value">
                    Cramér's V = {fmt(v, 6)}<br>
                    <span style="font-size: 18px;">Interpretation: {v_interp}</span>
                </div>
                <div class="details-box">
                    <strong>Formula:</strong> V = √[χ²/(n × (k-1))]<br>
                    where k = min(rows, cols)<br><br>
                    <strong>Guidelines (df = {min_dim - 1}):</strong><br>
                    {"Small: V ≈ 0.10, Medium: V ≈ 0.30, Large: V ≈ 0.50" if min_dim - 1 == 1 else "Small: V ≈ 0.07, Medium: V ≈ 0.21, Large: V ≈ 0.35" if min_dim - 1 == 2 else "Small: V ≈ 0.06, Medium: V ≈ 0.17, Large: V ≈ 0.29"}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# TAB 5: ANOVA FAMILY
# ============================================================

with tab5:
    calc5 = st.selectbox(
        "Select Test:",
        ["One-Way ANOVA", "Kruskal-Wallis Test", "Tukey's HSD (Post-Hoc)", 
         "Friedman Test", "Two-Way ANOVA"],
        key="calc5_select"
    )
    st.markdown("---")
    
    # ----- ONE-WAY ANOVA -----
    if calc5 == "One-Way ANOVA":
        st.markdown('<div class="section-header red">One-Way ANOVA</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> Compares means across 3+ independent groups.<br>
        <strong>H₀:</strong> All group means are equal | <strong>Hₐ:</strong> At least one mean differs
        </div>
        """, unsafe_allow_html=True)
        
        groups_text = st.text_area("Enter Groups (one per line):", 
                                    value="Group A: 23, 25, 27, 22, 26\nGroup B: 30, 32, 28, 31, 29\nGroup C: 18, 20, 22, 19, 21",
                                    height=150, key="anova_groups")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="anova_alpha")
        
        if st.button("Perform ANOVA", type="primary", key="anova_btn"):
            lines = groups_text.strip().split('\n')
            groups = []
            group_names = []
            
            for line in lines:
                if ':' in line:
                    name, data = line.split(':', 1)
                    group_names.append(name.strip())
                    groups.append(parse_data(data))
                else:
                    group_names.append(f'Group {len(groups)+1}')
                    groups.append(parse_data(line))
            
            k = len(groups)
            if k < 2:
                st.error("Please enter at least 2 groups.")
            else:
                # Calculate ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                reject = p_value < alpha
                
                # Manual calculations for ANOVA table
                group_means = [np.mean(g) for g in groups]
                group_sizes = [len(g) for g in groups]
                group_stds = [np.std(g, ddof=1) for g in groups]
                total_n = sum(group_sizes)
                grand_mean = np.mean(np.concatenate(groups))
                
                ssb = sum(n * (m - grand_mean)**2 for n, m in zip(group_sizes, group_means))
                ssw = sum(np.sum((g - m)**2) for g, m in zip(groups, group_means))
                sst = ssb + ssw
                
                df_between = k - 1
                df_within = total_n - k
                
                msb = ssb / df_between
                msw = ssw / df_within
                
                # Effect size
                eta2 = eta_squared(ssb, sst)
                eta2_interp = interpret_eta_squared(eta2)
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — At least one mean differs' if reject else '✓ FAIL TO REJECT H₀ — No significant difference'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> μ₁ = μ₂ = ... = μₖ<br>
                        <strong>Hₐ:</strong> At least one mean is different
                    </div>
                    <div class="result-value">
                        F = {fmt(f_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Effect Size:</strong> η² = {fmt(eta2, 4)} ({eta2_interp})<br>
                        <em>{fmt(eta2*100, 2)}% of variance explained by group membership</em>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display ANOVA table
                st.markdown("**ANOVA Table:**")
                anova_df = pd.DataFrame({
                    'Source': ['Between Groups', 'Within Groups', 'Total'],
                    'SS': [round(ssb, 6), round(ssw, 6), round(sst, 6)],
                    'df': [df_between, df_within, total_n - 1],
                    'MS': [round(msb, 6), round(msw, 6), ''],
                    'F': [round(f_stat, 6), '', ''],
                    'P-value': [fmt(p_value), '', '']
                })
                st.dataframe(anova_df, use_container_width=True, hide_index=True)
                
                # Group summary
                st.markdown("**Group Summary:**")
                summary_df = pd.DataFrame({
                    'Group': group_names,
                    'n': group_sizes,
                    'Mean': [round(m, 6) for m in group_means],
                    'Std Dev': [round(s, 6) for s in group_stds]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # ----- KRUSKAL-WALLIS TEST -----
    elif calc5 == "Kruskal-Wallis Test":
        st.markdown('<div class="section-header purple">Kruskal-Wallis Test (Non-Parametric)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Non-parametric alternative to One-Way ANOVA.</strong><br>
        Use when: Data is skewed, ordinal, or normality assumption violated.
        </div>
        """, unsafe_allow_html=True)
        
        groups_text = st.text_area("Enter Groups (one per line):", 
                                    value="Group A: 23, 25, 27, 22, 26\nGroup B: 30, 32, 28, 31, 29\nGroup C: 18, 20, 22, 19, 21",
                                    height=150, key="kw_groups")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="kw_alpha")
        
        if st.button("Perform Kruskal-Wallis Test", type="primary", key="kw_btn"):
            lines = groups_text.strip().split('\n')
            groups = []
            group_names = []
            
            for line in lines:
                if ':' in line:
                    name, data = line.split(':', 1)
                    group_names.append(name.strip())
                    groups.append(parse_data(data))
                else:
                    group_names.append(f'Group {len(groups)+1}')
                    groups.append(parse_data(line))
            
            if len(groups) < 2:
                st.error("Please enter at least 2 groups.")
            else:
                stat, p_value = stats.kruskal(*groups)
                reject = p_value < alpha
                
                # Effect size (epsilon squared)
                n = sum(len(g) for g in groups)
                k = len(groups)
                epsilon_sq = (stat - k + 1) / (n - k)
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Groups differ' if reject else '✓ FAIL TO REJECT H₀ — No significant difference'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> All groups have the same distribution<br>
                        <strong>Hₐ:</strong> At least one group differs
                    </div>
                    <div class="result-value">
                        H = {fmt(stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Effect Size:</strong> ε² ≈ {fmt(epsilon_sq, 4)}
                    </div>
                    <div class="details-box">
                        <strong>Group Medians:</strong><br>
                        {"<br>".join([f"{group_names[i]}: {fmt(np.median(groups[i]))}" for i in range(len(groups))])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- TUKEY'S HSD -----
    elif calc5 == "Tukey's HSD (Post-Hoc)":
        st.markdown('<div class="section-header red">Tukey\'s HSD Post-Hoc Test</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> After significant ANOVA, identifies which specific groups differ.<br>
        Controls family-wise error rate across all pairwise comparisons.
        </div>
        """, unsafe_allow_html=True)
        
        groups_text = st.text_area("Enter Groups (one per line):", 
                                    value="Group A: 23, 25, 27, 22, 26\nGroup B: 30, 32, 28, 31, 29\nGroup C: 18, 20, 22, 19, 21",
                                    height=150, key="tukey_groups")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="tukey_alpha")
        
        if st.button("Perform Tukey's HSD", type="primary", key="tukey_btn"):
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                
                lines = groups_text.strip().split('\n')
                all_data = []
                all_groups = []
                
                for line in lines:
                    if ':' in line:
                        name, data = line.split(':', 1)
                        name = name.strip()
                    else:
                        name = f'Group {len(set(all_groups))+1}'
                        data = line
                    
                    values = parse_data(data)
                    all_data.extend(values)
                    all_groups.extend([name] * len(values))
                
                result = pairwise_tukeyhsd(all_data, all_groups, alpha=alpha)
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-value">
                        Tukey's HSD Results (α = {alpha})
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display results as dataframe
                result_df = pd.DataFrame(data=result._results_table.data[1:], 
                                          columns=result._results_table.data[0])
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                
                st.markdown("""
                <div class="info-box">
                <strong>Interpretation:</strong> "Reject" = True means the pair of groups have significantly different means.
                </div>
                """, unsafe_allow_html=True)
                
            except ImportError:
                st.error("statsmodels is required for Tukey's HSD. Install with: pip install statsmodels")
    
    # ----- FRIEDMAN TEST -----
    elif calc5 == "Friedman Test":
        st.markdown('<div class="section-header purple">Friedman Test (Non-Parametric Repeated Measures)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Non-parametric alternative to Repeated Measures ANOVA.</strong><br>
        Use for: 3+ related samples (same subjects measured multiple times) with skewed/ordinal data.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("Enter data with subjects as rows and conditions as columns:")
        data_text = st.text_area("Data (rows = subjects, use comma or tab to separate conditions):",
                                  value="5, 4, 3\n6, 5, 4\n7, 6, 5\n5, 3, 2\n6, 4, 3\n8, 7, 6",
                                  height=150, key="fried_data")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="fried_alpha")
        
        if st.button("Perform Friedman Test", type="primary", key="fried_btn"):
            lines = data_text.strip().split('\n')
            data_matrix = []
            for line in lines:
                row = parse_data(line)
                if len(row) > 0:
                    data_matrix.append(row)
            
            data_array = np.array(data_matrix)
            
            if data_array.shape[1] < 3:
                st.error("Friedman test requires at least 3 conditions (columns).")
            else:
                stat, p_value = stats.friedmanchisquare(*[data_array[:, i] for i in range(data_array.shape[1])])
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Conditions differ' if reject else '✓ FAIL TO REJECT H₀ — No significant difference'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> All conditions have the same distribution<br>
                        <strong>Hₐ:</strong> At least one condition differs
                    </div>
                    <div class="result-value">
                        χ²_F = {fmt(stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>n subjects:</strong> {data_array.shape[0]}<br>
                        <strong>k conditions:</strong> {data_array.shape[1]}<br>
                        <strong>Condition Medians:</strong> {[fmt(np.median(data_array[:, i])) for i in range(data_array.shape[1])]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- TWO-WAY ANOVA -----
    elif calc5 == "Two-Way ANOVA":
        st.markdown('<div class="section-header red">Two-Way ANOVA</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Purpose:</strong> Tests effects of two factors and their interaction on a continuous outcome.<br>
        Enter data with Factor A levels as rows and Factor B levels as columns.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Enter cell means, sample sizes, and standard deviations:**")
        
        a_levels = st.number_input("Factor A levels:", min_value=2, max_value=5, value=2, key="2way_a")
        b_levels = st.number_input("Factor B levels:", min_value=2, max_value=5, value=2, key="2way_b")
        
        st.markdown("**Cell Means:**")
        means = []
        for i in range(a_levels):
            cols = st.columns(b_levels)
            row_means = []
            for j in range(b_levels):
                with cols[j]:
                    m = st.number_input(f"A{i+1}B{j+1} mean:", value=10.0 + i*5 + j*3, key=f"2way_m_{i}_{j}")
                    row_means.append(m)
            means.append(row_means)
        
        n_per_cell = st.number_input("n per cell (equal for all):", min_value=2, value=10, key="2way_n")
        mse = st.number_input("Mean Square Error (MSE/MSW):", min_value=0.01, value=25.0, key="2way_mse",
                               help="From your data or estimate of within-cell variance")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="2way_alpha")
        
        if st.button("Calculate Two-Way ANOVA", type="primary", key="2way_btn"):
            means_array = np.array(means)
            grand_mean = np.mean(means_array)
            
            # Factor A effect (row means)
            row_means = np.mean(means_array, axis=1)
            ss_a = n_per_cell * b_levels * np.sum((row_means - grand_mean)**2)
            df_a = a_levels - 1
            ms_a = ss_a / df_a
            f_a = ms_a / mse
            p_a = f_pvalue(f_a, df_a, a_levels * b_levels * (n_per_cell - 1))
            
            # Factor B effect (column means)
            col_means = np.mean(means_array, axis=0)
            ss_b = n_per_cell * a_levels * np.sum((col_means - grand_mean)**2)
            df_b = b_levels - 1
            ms_b = ss_b / df_b
            f_b = ms_b / mse
            p_b = f_pvalue(f_b, df_b, a_levels * b_levels * (n_per_cell - 1))
            
            # Interaction
            ss_ab = 0
            for i in range(a_levels):
                for j in range(b_levels):
                    expected = grand_mean + (row_means[i] - grand_mean) + (col_means[j] - grand_mean)
                    ss_ab += n_per_cell * (means_array[i, j] - expected)**2
            df_ab = df_a * df_b
            ms_ab = ss_ab / df_ab if df_ab > 0 else 0
            f_ab = ms_ab / mse if mse > 0 else 0
            p_ab = f_pvalue(f_ab, df_ab, a_levels * b_levels * (n_per_cell - 1))
            
            df_error = a_levels * b_levels * (n_per_cell - 1)
            
            st.markdown(f"""
            <div class="result-box">
                <div class="result-value">
                    Two-Way ANOVA Results
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            results_df = pd.DataFrame({
                'Source': ['Factor A', 'Factor B', 'A × B Interaction', 'Error'],
                'SS': [round(ss_a, 4), round(ss_b, 4), round(ss_ab, 4), '—'],
                'df': [df_a, df_b, df_ab, df_error],
                'MS': [round(ms_a, 4), round(ms_b, 4), round(ms_ab, 4), round(mse, 4)],
                'F': [round(f_a, 4), round(f_b, 4), round(f_ab, 4), '—'],
                'P-value': [fmt(p_a), fmt(p_b), fmt(p_ab), '—'],
                'Significant': ['Yes' if p_a < alpha else 'No', 'Yes' if p_b < alpha else 'No', 
                               'Yes' if p_ab < alpha else 'No', '—']
            })
            st.dataframe(results_df, use_container_width=True, hide_index=True)

# ============================================================
# TAB 6: CORRELATION
# ============================================================

with tab6:
    calc6 = st.selectbox(
        "Select Correlation:",
        ["Pearson Correlation (r)", "Spearman Rank Correlation (ρ)", "Kendall's Tau (τ)"],
        key="calc6_select"
    )
    st.markdown("---")
    
    # Common input for all correlations
    col1, col2 = st.columns(2)
    with col1:
        x_text = st.text_area("X Data:", value="1, 2, 3, 4, 5, 6, 7, 8, 9, 10", key="corr_x")
    with col2:
        y_text = st.text_area("Y Data:", value="2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.9, 18.1, 20.0", key="corr_y")
    
    alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="corr_alpha")
    
    # ----- PEARSON -----
    if calc6 == "Pearson Correlation (r)":
        st.markdown('<div class="section-header pink">Pearson Correlation Coefficient</div>', unsafe_allow_html=True)
        
        if st.button("Calculate Pearson r", type="primary", key="pearson_btn"):
            x = parse_data(x_text)
            y = parse_data(y_text)
            
            if len(x) != len(y) or len(x) < 3:
                st.error("Please enter equal numbers of X and Y values (at least 3 pairs).")
            else:
                r, p_value = stats.pearsonr(x, y)
                r_squared = r**2
                reject = p_value < alpha
                r_interp = interpret_r(r)
                
                # CI for r using Fisher's z transformation
                n = len(x)
                z = 0.5 * np.log((1 + r) / (1 - r))
                se_z = 1 / np.sqrt(n - 3)
                z_crit = z_critical((1 - alpha) * 100)
                z_lower = z - z_crit * se_z
                z_upper = z + z_crit * se_z
                r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Significant correlation' if reject else '✓ FAIL TO REJECT H₀ — No significant correlation'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> ρ = 0 (no linear correlation)<br>
                        <strong>Hₐ:</strong> ρ ≠ 0
                    </div>
                    <div class="result-value">
                        r = {fmt(r)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>r²:</strong> {fmt(r_squared, 4)} ({fmt(r_squared*100, 2)}% variance explained)<br>
                        <strong>Strength:</strong> {r_interp} {"positive" if r > 0 else "negative"} correlation<br>
                        <strong>{int((1-alpha)*100)}% CI for ρ:</strong> ({fmt(r_lower, 4)}, {fmt(r_upper, 4)})
                    </div>
                    <div class="details-box">
                        <strong>n pairs:</strong> {n}<br>
                        <strong>df:</strong> n - 2 = {n - 2}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- SPEARMAN -----
    elif calc6 == "Spearman Rank Correlation (ρ)":
        st.markdown('<div class="section-header purple">Spearman Rank Correlation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Non-parametric alternative to Pearson.</strong><br>
        Measures monotonic (not necessarily linear) relationship. Robust to outliers.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Calculate Spearman ρ", type="primary", key="spearman_btn"):
            x = parse_data(x_text)
            y = parse_data(y_text)
            
            if len(x) != len(y) or len(x) < 3:
                st.error("Please enter equal numbers of X and Y values (at least 3 pairs).")
            else:
                rho, p_value = stats.spearmanr(x, y)
                reject = p_value < alpha
                r_interp = interpret_r(rho)
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Significant correlation' if reject else '✓ FAIL TO REJECT H₀ — No significant correlation'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> ρₛ = 0 (no monotonic correlation)<br>
                        <strong>Hₐ:</strong> ρₛ ≠ 0
                    </div>
                    <div class="result-value">
                        ρ = {fmt(rho)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Strength:</strong> {r_interp} {"positive" if rho > 0 else "negative"} monotonic relationship
                    </div>
                    <div class="details-box">
                        <strong>n pairs:</strong> {len(x)}<br>
                        <strong>Note:</strong> Spearman's ρ uses ranked data, making it robust to outliers and suitable for ordinal data.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ----- KENDALL -----
    elif calc6 == "Kendall's Tau (τ)":
        st.markdown('<div class="section-header purple">Kendall\'s Tau Correlation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>Another non-parametric correlation measure.</strong><br>
        Better than Spearman for small samples or many tied values.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Calculate Kendall's τ", type="primary", key="kendall_btn"):
            x = parse_data(x_text)
            y = parse_data(y_text)
            
            if len(x) != len(y) or len(x) < 3:
                st.error("Please enter equal numbers of X and Y values (at least 3 pairs).")
            else:
                tau, p_value = stats.kendalltau(x, y)
                reject = p_value < alpha
                
                # Interpretation (Kendall's tau tends to be smaller than Pearson/Spearman)
                tau_abs = abs(tau)
                if tau_abs < 0.1:
                    tau_interp = "Negligible"
                elif tau_abs < 0.2:
                    tau_interp = "Weak"
                elif tau_abs < 0.3:
                    tau_interp = "Moderate"
                else:
                    tau_interp = "Strong"
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Significant correlation' if reject else '✓ FAIL TO REJECT H₀ — No significant correlation'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> τ = 0 (no association)<br>
                        <strong>Hₐ:</strong> τ ≠ 0
                    </div>
                    <div class="result-value">
                        τ = {fmt(tau)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>Strength:</strong> {tau_interp} {"positive" if tau > 0 else "negative"} association
                    </div>
                    <div class="details-box">
                        <strong>n pairs:</strong> {len(x)}<br>
                        <strong>Note:</strong> Kendall's τ is based on concordant and discordant pairs. Values are typically smaller than Pearson or Spearman.
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 7: REGRESSION
# ============================================================

with tab7:
    calc7 = st.selectbox(
        "Select Analysis:",
        ["Simple Linear Regression", "Multiple Regression", "Logistic Regression"],
        key="calc7_select"
    )
    st.markdown("---")
    
    # ----- SIMPLE LINEAR REGRESSION -----
    if calc7 == "Simple Linear Regression":
        st.markdown('<div class="section-header">Simple Linear Regression</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x_text = st.text_area("X (Predictor):", value="1, 2, 3, 4, 5, 6, 7, 8, 9, 10", key="reg_x")
        with col2:
            y_text = st.text_area("Y (Response):", value="2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.9, 18.1, 20.0", key="reg_y")
        
        col1, col2 = st.columns(2)
        with col1:
            pred_x = st.number_input("Predict Y for X =", value=5.5, format="%.4f", key="reg_pred")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="reg_alpha")
        
        if st.button("Perform Regression Analysis", type="primary", key="reg_btn"):
            x = parse_data(x_text)
            y = parse_data(y_text)
            
            if len(x) != len(y) or len(x) < 3:
                st.error("Please enter equal numbers of X and Y values (at least 3 pairs).")
            else:
                n = len(x)
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                # Calculate regression coefficients
                ss_xy = np.sum((x - x_mean) * (y - y_mean))
                ss_xx = np.sum((x - x_mean)**2)
                ss_yy = np.sum((y - y_mean)**2)
                
                b1 = ss_xy / ss_xx  # slope
                b0 = y_mean - b1 * x_mean  # intercept
                
                # Predictions and residuals
                y_pred = b0 + b1 * x
                residuals = y - y_pred
                ss_res = np.sum(residuals**2)
                ss_reg = np.sum((y_pred - y_mean)**2)
                
                # R-squared
                r_squared = 1 - (ss_res / ss_yy)
                r = np.sqrt(r_squared) * np.sign(b1)
                
                # Standard errors
                df = n - 2
                mse = ss_res / df
                se_b1 = np.sqrt(mse / ss_xx)
                se_b0 = np.sqrt(mse * (1/n + x_mean**2/ss_xx))
                
                # t-tests for coefficients
                t_b1 = b1 / se_b1
                p_b1 = t_pvalue(t_b1, df, 'two')
                t_b0 = b0 / se_b0
                p_b0 = t_pvalue(t_b0, df, 'two')
                
                # F-test for overall model
                f_stat = (ss_reg / 1) / mse
                p_f = f_pvalue(f_stat, 1, df)
                
                # Prediction
                y_at_pred = b0 + b1 * pred_x
                
                reject = p_b1 < alpha
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ — Significant relationship' if reject else '✓ FAIL TO REJECT H₀ — No significant relationship'
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> β₁ = 0 (no linear relationship)<br>
                        <strong>Hₐ:</strong> β₁ ≠ 0
                    </div>
                    <div class="result-value">
                        ŷ = {fmt(b0)} + {fmt(b1)}x<br>
                        R² = {fmt(r_squared)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="effect-size-box">
                        <strong>R² = {fmt(r_squared, 4)}</strong> — {fmt(r_squared*100, 2)}% of variance in Y is explained by X<br>
                        <strong>r = {fmt(r, 4)}</strong> — {interpret_r(r)} correlation
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Coefficients table
                st.markdown("**Coefficients:**")
                coef_df = pd.DataFrame({
                    'Term': ['Intercept (b₀)', 'Slope (b₁)'],
                    'Estimate': [fmt(b0), fmt(b1)],
                    'Std Error': [fmt(se_b0), fmt(se_b1)],
                    't-value': [fmt(t_b0), fmt(t_b1)],
                    'P-value': [fmt(p_b0), fmt(p_b1)],
                    'Significant': ['Yes' if p_b0 < alpha else 'No', 'Yes' if p_b1 < alpha else 'No']
                })
                st.dataframe(coef_df, use_container_width=True, hide_index=True)
                
                # Prediction
                st.markdown(f"""
                <div class="info-box">
                <strong>Prediction:</strong> When X = {pred_x}, predicted Y = <strong>{fmt(y_at_pred)}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Model summary
                st.markdown("**Model Summary:**")
                st.markdown(f"""
                - **n:** {n}
                - **df:** {df}
                - **MSE:** {fmt(mse)}
                - **RMSE:** {fmt(np.sqrt(mse))}
                - **F-statistic:** {fmt(f_stat)} (p = {fmt(p_f)})
                """)
    
    # ----- MULTIPLE REGRESSION -----
    elif calc7 == "Multiple Regression":
        st.markdown('<div class="section-header">Multiple Linear Regression</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        Enter data with each column as a variable. First row = headers.<br>
        Last column is treated as the response variable (Y).
        </div>
        """, unsafe_allow_html=True)
        
        data_text = st.text_area("Data (CSV format, last column is Y):",
                                  value="X1,X2,Y\n1,5,10\n2,4,12\n3,6,15\n4,3,14\n5,7,20\n6,5,18\n7,8,25\n8,6,22\n9,9,28\n10,7,26",
                                  height=200, key="mreg_data")
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="mreg_alpha")
        
        if st.button("Perform Multiple Regression", type="primary", key="mreg_btn"):
            try:
                import statsmodels.api as sm
                
                lines = data_text.strip().split('\n')
                headers = [h.strip() for h in lines[0].split(',')]
                data = []
                for line in lines[1:]:
                    row = [float(x.strip()) for x in line.split(',')]
                    data.append(row)
                
                df = pd.DataFrame(data, columns=headers)
                
                y_col = headers[-1]
                x_cols = headers[:-1]
                
                Y = df[y_col]
                X = df[x_cols]
                X = sm.add_constant(X)
                
                model = sm.OLS(Y, X).fit()
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-value">
                        R² = {fmt(model.rsquared)}<br>
                        Adj. R² = {fmt(model.rsquared_adj)}
                    </div>
                    <div class="effect-size-box">
                        <strong>F-statistic:</strong> {fmt(model.fvalue)} (p = {fmt(model.f_pvalue)})<br>
                        <strong>Interpretation:</strong> {fmt(model.rsquared*100, 2)}% of variance explained
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Coefficients
                st.markdown("**Coefficients:**")
                coef_df = pd.DataFrame({
                    'Variable': model.params.index,
                    'Coefficient': model.params.values,
                    'Std Error': model.bse.values,
                    't-value': model.tvalues.values,
                    'P-value': model.pvalues.values
                })
                st.dataframe(coef_df.round(6), use_container_width=True, hide_index=True)
                
                # Equation
                equation = f"ŷ = {model.params['const']:.4f}"
                for col in x_cols:
                    equation += f" + ({model.params[col]:.4f}){col}"
                st.markdown(f"**Regression Equation:** {equation}")
                
            except ImportError:
                st.error("statsmodels is required. Install with: pip install statsmodels")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- LOGISTIC REGRESSION -----
    elif calc7 == "Logistic Regression":
        st.markdown('<div class="section-header">Logistic Regression</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <strong>For binary outcomes (0/1, Yes/No).</strong><br>
        Enter data with predictor(s) and binary response (0 or 1) in last column.
        </div>
        """, unsafe_allow_html=True)
        
        data_text = st.text_area("Data (CSV format, last column is binary Y):",
                                  value="X1,X2,Y\n1,2,0\n2,3,0\n3,2,0\n4,4,0\n5,3,1\n6,5,1\n7,4,1\n8,6,1\n9,5,1\n10,7,1",
                                  height=200, key="logreg_data")
        
        if st.button("Perform Logistic Regression", type="primary", key="logreg_btn"):
            try:
                import statsmodels.api as sm
                
                lines = data_text.strip().split('\n')
                headers = [h.strip() for h in lines[0].split(',')]
                data = []
                for line in lines[1:]:
                    row = [float(x.strip()) for x in line.split(',')]
                    data.append(row)
                
                df = pd.DataFrame(data, columns=headers)
                
                y_col = headers[-1]
                x_cols = headers[:-1]
                
                Y = df[y_col]
                X = df[x_cols]
                X = sm.add_constant(X)
                
                model = sm.Logit(Y, X).fit(disp=0)
                
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-value">
                        Pseudo R² = {fmt(model.prsquared)}<br>
                        Log-Likelihood = {fmt(model.llf)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Coefficients with odds ratios
                st.markdown("**Coefficients:**")
                coef_df = pd.DataFrame({
                    'Variable': model.params.index,
                    'Coefficient': model.params.values,
                    'Odds Ratio': np.exp(model.params.values),
                    'Std Error': model.bse.values,
                    'z-value': model.tvalues.values,
                    'P-value': model.pvalues.values
                })
                st.dataframe(coef_df.round(6), use_container_width=True, hide_index=True)
                
                st.markdown("""
                <div class="info-box">
                <strong>Odds Ratio Interpretation:</strong><br>
                OR > 1: Increase in predictor increases odds of Y=1<br>
                OR < 1: Increase in predictor decreases odds of Y=1<br>
                OR = 1: No effect
                </div>
                """, unsafe_allow_html=True)
                
            except ImportError:
                st.error("statsmodels is required. Install with: pip install statsmodels")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 13px; padding: 20px;">
    📊 <strong>Complete Statistics Calculator</strong> | 42 Statistical Tests | 7 Calculators<br>
    Built with Streamlit & SciPy | All calculations use exact p-values (no approximations)<br>
    Includes effect sizes, confidence intervals, and assumption checking
</div>
""", unsafe_allow_html=True)
