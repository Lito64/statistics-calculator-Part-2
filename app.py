import streamlit as st
import numpy as np
from scipy import stats
from decimal import Decimal, getcontext

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Statistics Calculator",
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 8px 0 0 0;
        opacity: 0.9;
        font-size: 16px;
    }
    
    .result-box-ci {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    
    .result-box-ht {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #f5576c;
        margin: 15px 0;
    }
    
    .result-value {
        font-size: 28px;
        font-weight: 700;
        color: #333;
        text-align: center;
        padding: 20px;
        background: white;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    
    .decision-reject {
        font-size: 22px;
        font-weight: 700;
        padding: 18px;
        background: white;
        border-radius: 10px;
        text-align: center;
        margin: 15px 0;
        color: #c62828;
        border: 3px solid #c62828;
    }
    
    .decision-fail {
        font-size: 22px;
        font-weight: 700;
        padding: 18px;
        background: white;
        border-radius: 10px;
        text-align: center;
        margin: 15px 0;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    
    .hypothesis-box {
        background: #fff3e0;
        padding: 15px 18px;
        border-radius: 10px;
        margin: 12px 0;
        border-left: 5px solid #ff9800;
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
    
    .section-header-ci {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 20px;
        font-weight: 600;
    }
    
    .section-header-ht {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 20px;
        font-weight: 600;
    }
    
    .section-header-adv {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 20px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PRECISION SETTINGS
# ============================================================

# Sidebar for settings
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    PRECISION = st.slider("Decimal Precision", min_value=4, max_value=16, value=8, step=1)
    getcontext().prec = PRECISION + 10
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This calculator provides high-precision 
    statistical computations using Python's 
    `scipy.stats` library.
    
    **Features:**
    - Exact p-values (no approximations)
    - Adjustable decimal precision
    - Step-by-step calculations
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Calculators")
    st.markdown("""
    **Confidence Intervals:**
    - One Mean, One Proportion
    - Two Means, Paired Samples
    - Two Proportions
    
    **Hypothesis Tests:**
    - t-tests, z-tests
    - Two-sample tests
    
    **Advanced:**
    - Chi-Square tests
    - ANOVA
    - Correlation & Regression
    """)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt(value, decimals=None):
    """Format number with specified precision."""
    if decimals is None:
        decimals = PRECISION
    if isinstance(value, (int, float, np.floating)):
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
    return np.std(data, ddof=1)

def sample_mean(data):
    return np.mean(data)

def t_critical(df, confidence_level):
    alpha = 1 - confidence_level / 100
    return stats.t.ppf(1 - alpha / 2, df)

def z_critical(confidence_level):
    alpha = 1 - confidence_level / 100
    return stats.norm.ppf(1 - alpha / 2)

def t_pvalue(t_stat, df, test_type='two'):
    if test_type == 'two':
        return 2 * stats.t.sf(abs(t_stat), df)
    elif test_type == 'left':
        return stats.t.cdf(t_stat, df)
    else:
        return stats.t.sf(t_stat, df)

def z_pvalue(z_stat, test_type='two'):
    if test_type == 'two':
        return 2 * stats.norm.sf(abs(z_stat))
    elif test_type == 'left':
        return stats.norm.cdf(z_stat)
    else:
        return stats.norm.sf(z_stat)

def chi2_pvalue(chi2_stat, df):
    return stats.chi2.sf(chi2_stat, df)

def f_pvalue(f_stat, df1, df2):
    return stats.f.sf(f_stat, df1, df2)

def welch_df(s1, n1, s2, n2):
    v1 = s1**2 / n1
    v2 = s2**2 / n2
    numerator = (v1 + v2)**2
    denominator = (v1**2 / (n1 - 1)) + (v2**2 / (n2 - 1))
    return numerator / denominator

# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>📊 Comprehensive Statistics Calculator</h1>
    <p>High-Precision Confidence Intervals • Hypothesis Tests • Advanced Analysis</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN TABS
# ============================================================

tab_ci, tab_ht, tab_adv = st.tabs(["📊 Confidence Intervals", "🧪 Hypothesis Tests", "🔬 Advanced Tests"])

# ============================================================
# TAB 1: CONFIDENCE INTERVALS
# ============================================================

with tab_ci:
    ci_subtab = st.selectbox(
        "Select Calculator:",
        ["One Mean (t-interval)", "One Proportion (z-interval)", "Two Independent Means", 
         "Paired Samples", "Two Proportions"],
        key="ci_select"
    )
    
    st.markdown("---")
    
    # ----- CI: ONE MEAN -----
    if ci_subtab == "One Mean (t-interval)":
        st.markdown('<div class="section-header-ci">Confidence Interval for μ (One Sample Mean)</div>', unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="ci1_input")
        
        col1, col2 = st.columns(2)
        
        if input_type == "Summary Statistics":
            with col1:
                n = st.number_input("Sample Size (n):", min_value=2, value=30, key="ci1_n")
                x_bar = st.number_input("Sample Mean (x̄):", value=0.0, format="%.6f", key="ci1_mean")
            with col2:
                s = st.number_input("Sample Std Dev (s):", min_value=0.0001, value=1.0, format="%.6f", key="ci1_std")
                cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci1_cl")
        else:
            raw_data = st.text_area("Enter data (comma or space separated):", 
                                     placeholder="12.5, 13.2, 11.8, 14.1, 12.9", key="ci1_raw")
            cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci1_cl_raw")
        
        if st.button("Calculate Confidence Interval", type="primary", key="ci1_btn"):
            try:
                if input_type == "Raw Data":
                    data = parse_data(raw_data)
                    if len(data) < 2:
                        st.error("Please enter at least 2 data values.")
                        st.stop()
                    n = len(data)
                    x_bar = sample_mean(data)
                    s = sample_std(data)
                
                df = n - 1
                t_star = t_critical(df, cl)
                se = s / np.sqrt(n)
                me = t_star * se
                lower = x_bar - me
                upper = x_bar + me
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="result-value">
                        ({fmt(lower)}, {fmt(upper)})
                    </div>
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        Sample size: n = {n}<br>
                        Sample mean: x̄ = {fmt(x_bar)}<br>
                        Sample std dev: s = {fmt(s)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        Degrees of Freedom: df = n - 1 = {df}<br>
                        Critical Value: t* = {fmt(t_star)} (for {cl}% CI)<br>
                        Standard Error: SE = s/√n = {fmt(s)}/√{n} = {fmt(se)}<br>
                        Margin of Error: ME = t* × SE = {fmt(t_star)} × {fmt(se)} = {fmt(me)}<br><br>
                        
                        <strong>Formula:</strong> CI = x̄ ± t* × (s/√n)<br><br>
                        
                        <strong>Interpretation:</strong> We are {cl}% confident that the true population mean μ lies between {fmt(lower)} and {fmt(upper)}.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- CI: ONE PROPORTION -----
    elif ci_subtab == "One Proportion (z-interval)":
        st.markdown('<div class="section-header-ci">Confidence Interval for p (One Proportion)</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("Number of Successes (x):", min_value=0, value=50, key="ci2_x")
            n = st.number_input("Sample Size (n):", min_value=1, value=100, key="ci2_n")
        with col2:
            cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci2_cl")
        
        if st.button("Calculate Confidence Interval", type="primary", key="ci2_btn"):
            try:
                if x > n:
                    st.error("Successes cannot exceed sample size.")
                    st.stop()
                
                p_hat = x / n
                z_star = z_critical(cl)
                se = np.sqrt(p_hat * (1 - p_hat) / n)
                me = z_star * se
                lower = max(0, p_hat - me)
                upper = min(1, p_hat + me)
                
                cond_met = (n * p_hat >= 10) and (n * (1 - p_hat) >= 10)
                cond_text = '✓ Conditions met (np̂ ≥ 10 and n(1-p̂) ≥ 10)' if cond_met else '⚠️ Warning: Conditions not met'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="result-value">
                        ({fmt(lower)}, {fmt(upper)})<br>
                        <span style="font-size:18px;">or ({fmt(lower*100, 2)}%, {fmt(upper*100, 2)}%)</span>
                    </div>
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        Successes: x = {x}<br>
                        Sample size: n = {n}<br>
                        Sample proportion: p̂ = x/n = {x}/{n} = {fmt(p_hat)}<br><br>
                        
                        <strong>Conditions Check:</strong> {cond_text}<br>
                        np̂ = {fmt(n * p_hat, 1)}, n(1-p̂) = {fmt(n * (1 - p_hat), 1)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        Critical Value: z* = {fmt(z_star)} (for {cl}% CI)<br>
                        Standard Error: SE = √[p̂(1-p̂)/n] = {fmt(se)}<br>
                        Margin of Error: ME = z* × SE = {fmt(me)}<br><br>
                        
                        <strong>Formula:</strong> CI = p̂ ± z* × √[p̂(1-p̂)/n]<br><br>
                        
                        <strong>Interpretation:</strong> We are {cl}% confident that the true population proportion p lies between {fmt(lower)} and {fmt(upper)} (or {fmt(lower*100, 2)}% to {fmt(upper*100, 2)}%).
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- CI: TWO INDEPENDENT MEANS -----
    elif ci_subtab == "Two Independent Means":
        st.markdown('<div class="section-header-ci">Confidence Interval for μ₁ - μ₂</div>', unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="ci3_input")
        
        if input_type == "Summary Statistics":
            st.markdown("**Group 1:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                n1 = st.number_input("n₁:", min_value=2, value=30, key="ci3_n1")
            with col2:
                x1 = st.number_input("x̄₁:", value=0.0, format="%.6f", key="ci3_x1")
            with col3:
                s1 = st.number_input("s₁:", min_value=0.0001, value=1.0, format="%.6f", key="ci3_s1")
            
            st.markdown("**Group 2:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                n2 = st.number_input("n₂:", min_value=2, value=30, key="ci3_n2")
            with col2:
                x2 = st.number_input("x̄₂:", value=0.0, format="%.6f", key="ci3_x2")
            with col3:
                s2 = st.number_input("s₂:", min_value=0.0001, value=1.0, format="%.6f", key="ci3_s2")
            
            cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci3_cl")
        else:
            col1, col2 = st.columns(2)
            with col1:
                raw1 = st.text_area("Group 1 Data:", placeholder="Enter values...", key="ci3_raw1")
            with col2:
                raw2 = st.text_area("Group 2 Data:", placeholder="Enter values...", key="ci3_raw2")
            cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci3_cl_raw")
        
        if st.button("Calculate Confidence Interval", type="primary", key="ci3_btn"):
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
                
                se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
                df = welch_df(s1, n1, s2, n2)
                t_star = t_critical(df, cl)
                diff = x1 - x2
                me = t_star * se
                lower = diff - me
                upper = diff + me
                
                if lower > 0:
                    interp = '→ Group 1 mean is significantly higher'
                elif upper < 0:
                    interp = '→ Group 2 mean is significantly higher'
                else:
                    interp = '→ No significant difference (interval contains 0)'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="result-value">
                        ({fmt(lower)}, {fmt(upper)})
                    </div>
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        Group 1: n₁ = {n1}, x̄₁ = {fmt(x1)}, s₁ = {fmt(s1)}<br>
                        Group 2: n₂ = {n2}, x̄₂ = {fmt(x2)}, s₂ = {fmt(s2)}<br>
                        Difference: x̄₁ - x̄₂ = {fmt(diff)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        Standard Error: SE = √[(s₁²/n₁) + (s₂²/n₂)] = {fmt(se)}<br>
                        Degrees of Freedom: df ≈ {fmt(df, 2)} (Welch's approximation)<br>
                        Critical Value: t* = {fmt(t_star)} (for {cl}% CI)<br>
                        Margin of Error: ME = t* × SE = {fmt(me)}<br><br>
                        
                        <strong>Formula:</strong> CI = (x̄₁ - x̄₂) ± t* × √[(s₁²/n₁) + (s₂²/n₂)]<br><br>
                        
                        <strong>Interpretation:</strong> We are {cl}% confident that the true difference (μ₁ - μ₂) lies between {fmt(lower)} and {fmt(upper)}.<br>
                        {interp}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- CI: PAIRED SAMPLES -----
    elif ci_subtab == "Paired Samples":
        st.markdown('<div class="section-header-ci">Confidence Interval for μ_d (Paired Samples)</div>', unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="ci4_input")
        
        if input_type == "Summary Statistics":
            col1, col2 = st.columns(2)
            with col1:
                n = st.number_input("Number of Pairs (n):", min_value=2, value=30, key="ci4_n")
                d_bar = st.number_input("Mean Difference (d̄):", value=0.0, format="%.6f", key="ci4_dbar")
            with col2:
                s_d = st.number_input("Std Dev of Differences (s_d):", min_value=0.0001, value=1.0, format="%.6f", key="ci4_sd")
                cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci4_cl")
        else:
            col1, col2 = st.columns(2)
            with col1:
                before = st.text_area("Before/Group 1:", placeholder="Enter values...", key="ci4_before")
            with col2:
                after = st.text_area("After/Group 2:", placeholder="Enter values...", key="ci4_after")
            cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci4_cl_raw")
        
        if st.button("Calculate Confidence Interval", type="primary", key="ci4_btn"):
            try:
                if input_type == "Raw Data":
                    before_data = parse_data(before)
                    after_data = parse_data(after)
                    if len(before_data) != len(after_data) or len(before_data) < 2:
                        st.error("Please enter equal numbers of values (at least 2 pairs).")
                        st.stop()
                    diff = before_data - after_data
                    n = len(diff)
                    d_bar = sample_mean(diff)
                    s_d = sample_std(diff)
                
                df = n - 1
                t_star = t_critical(df, cl)
                se = s_d / np.sqrt(n)
                me = t_star * se
                lower = d_bar - me
                upper = d_bar + me
                
                if lower > 0:
                    interp = '→ Significant positive difference (Before > After)'
                elif upper < 0:
                    interp = '→ Significant negative difference (Before < After)'
                else:
                    interp = '→ No significant difference (interval contains 0)'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="result-value">
                        ({fmt(lower)}, {fmt(upper)})
                    </div>
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        Number of pairs: n = {n}<br>
                        Mean difference: d̄ = {fmt(d_bar)}<br>
                        Std dev of differences: s_d = {fmt(s_d)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        Degrees of Freedom: df = n - 1 = {df}<br>
                        Critical Value: t* = {fmt(t_star)} (for {cl}% CI)<br>
                        Standard Error: SE = s_d/√n = {fmt(s_d)}/√{n} = {fmt(se)}<br>
                        Margin of Error: ME = t* × SE = {fmt(me)}<br><br>
                        
                        <strong>Formula:</strong> CI = d̄ ± t* × (s_d/√n)<br><br>
                        
                        <strong>Interpretation:</strong> We are {cl}% confident that the true mean difference μ_d lies between {fmt(lower)} and {fmt(upper)}.<br>
                        {interp}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- CI: TWO PROPORTIONS -----
    elif ci_subtab == "Two Proportions":
        st.markdown('<div class="section-header-ci">Confidence Interval for p₁ - p₂</div>', unsafe_allow_html=True)
        
        st.markdown("**Group 1:**")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("Successes (x₁):", min_value=0, value=50, key="ci5_x1")
        with col2:
            n1 = st.number_input("Sample Size (n₁):", min_value=1, value=100, key="ci5_n1")
        
        st.markdown("**Group 2:**")
        col1, col2 = st.columns(2)
        with col1:
            x2 = st.number_input("Successes (x₂):", min_value=0, value=40, key="ci5_x2")
        with col2:
            n2 = st.number_input("Sample Size (n₂):", min_value=1, value=100, key="ci5_n2")
        
        cl = st.selectbox("Confidence Level:", [90, 95, 99], index=1, key="ci5_cl")
        
        if st.button("Calculate Confidence Interval", type="primary", key="ci5_btn"):
            try:
                p1 = x1 / n1
                p2 = x2 / n2
                diff = p1 - p2
                
                cond_met = all([n1*p1 >= 10, n1*(1-p1) >= 10, n2*p2 >= 10, n2*(1-p2) >= 10])
                cond_text = '✓ All conditions met' if cond_met else '⚠️ Warning: Some conditions not met'
                
                se = np.sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
                z_star = z_critical(cl)
                me = z_star * se
                lower = diff - me
                upper = diff + me
                
                if lower > 0:
                    interp = '→ Group 1 proportion is significantly higher'
                elif upper < 0:
                    interp = '→ Group 2 proportion is significantly higher'
                else:
                    interp = '→ No significant difference (interval contains 0)'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="result-value">
                        ({fmt(lower)}, {fmt(upper)})<br>
                        <span style="font-size:18px;">or ({fmt(lower*100, 2)}%, {fmt(upper*100, 2)}%)</span>
                    </div>
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        Group 1: x₁ = {x1}, n₁ = {n1}, p̂₁ = {fmt(p1)}<br>
                        Group 2: x₂ = {x2}, n₂ = {n2}, p̂₂ = {fmt(p2)}<br>
                        Difference: p̂₁ - p̂₂ = {fmt(diff)}<br><br>
                        
                        <strong>Conditions Check:</strong> {cond_text}<br>
                        n₁p̂₁ = {fmt(n1*p1, 1)}, n₁(1-p̂₁) = {fmt(n1*(1-p1), 1)}<br>
                        n₂p̂₂ = {fmt(n2*p2, 1)}, n₂(1-p̂₂) = {fmt(n2*(1-p2), 1)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        Standard Error: SE = √[(p̂₁(1-p̂₁)/n₁) + (p̂₂(1-p̂₂)/n₂)] = {fmt(se)}<br>
                        Critical Value: z* = {fmt(z_star)} (for {cl}% CI)<br>
                        Margin of Error: ME = z* × SE = {fmt(me)}<br><br>
                        
                        <strong>Formula:</strong> CI = (p̂₁ - p̂₂) ± z* × √[(p̂₁(1-p̂₁)/n₁) + (p̂₂(1-p̂₂)/n₂)]<br><br>
                        
                        <strong>Interpretation:</strong> We are {cl}% confident that the true difference (p₁ - p₂) lies between {fmt(lower)} and {fmt(upper)}.<br>
                        {interp}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================
# TAB 2: HYPOTHESIS TESTS
# ============================================================

with tab_ht:
    ht_subtab = st.selectbox(
        "Select Calculator:",
        ["One Mean (t-test)", "One Proportion (z-test)", "Two Independent Means", 
         "Paired Samples", "Two Proportions"],
        key="ht_select"
    )
    
    st.markdown("---")
    
    # ----- HT: ONE MEAN -----
    if ht_subtab == "One Mean (t-test)":
        st.markdown('<div class="section-header-ht">One Sample t-Test</div>', unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="ht1_input")
        
        col1, col2 = st.columns(2)
        
        if input_type == "Summary Statistics":
            with col1:
                n = st.number_input("Sample Size (n):", min_value=2, value=30, key="ht1_n")
                x_bar = st.number_input("Sample Mean (x̄):", value=0.0, format="%.6f", key="ht1_mean")
                s = st.number_input("Sample Std Dev (s):", min_value=0.0001, value=1.0, format="%.6f", key="ht1_std")
            with col2:
                mu0 = st.number_input("Null Value (μ₀):", value=0.0, format="%.6f", key="ht1_mu0")
                test_type = st.selectbox("Test Type:", 
                    ["Two-tailed (μ ≠ μ₀)", "Left-tailed (μ < μ₀)", "Right-tailed (μ > μ₀)"], key="ht1_type")
                alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="ht1_alpha")
        else:
            with col1:
                raw_data = st.text_area("Enter data:", placeholder="12.5, 13.2, 11.8...", key="ht1_raw")
            with col2:
                mu0 = st.number_input("Null Value (μ₀):", value=0.0, format="%.6f", key="ht1_mu0_raw")
                test_type = st.selectbox("Test Type:", 
                    ["Two-tailed (μ ≠ μ₀)", "Left-tailed (μ < μ₀)", "Right-tailed (μ > μ₀)"], key="ht1_type_raw")
                alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="ht1_alpha_raw")
        
        if st.button("Perform Hypothesis Test", type="primary", key="ht1_btn"):
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
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                conclusion = f"There IS sufficient evidence at α = {alpha} to conclude that μ {symbols[tt]} {mu0}." if reject else f"There is NOT sufficient evidence at α = {alpha} to conclude that μ {symbols[tt]} {mu0}."
                
                st.markdown(f"""
                <div class="result-box-ht">
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
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        Sample size: n = {n}<br>
                        Sample mean: x̄ = {fmt(x_bar)}<br>
                        Sample std dev: s = {fmt(s)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        Degrees of Freedom: df = n - 1 = {df}<br>
                        Standard Error: SE = s/√n = {fmt(s)}/√{n} = {fmt(se)}<br>
                        Test Statistic: t = (x̄ - μ₀)/SE = ({fmt(x_bar)} - {mu0})/{fmt(se)} = {fmt(t_stat)}<br>
                        P-value = {fmt(p_value)}<br>
                        Significance level: α = {alpha}<br><br>
                        
                        <strong>Decision Rule:</strong> {"P-value < α" if reject else "P-value ≥ α"} → {"Reject H₀" if reject else "Fail to reject H₀"}<br><br>
                        
                        <strong>Conclusion:</strong> {conclusion}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- HT: ONE PROPORTION -----
    elif ht_subtab == "One Proportion (z-test)":
        st.markdown('<div class="section-header-ht">One Sample z-Test for Proportion</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input("Number of Successes (x):", min_value=0, value=50, key="ht2_x")
            n = st.number_input("Sample Size (n):", min_value=1, value=100, key="ht2_n")
        with col2:
            p0 = st.number_input("Null Value (p₀):", min_value=0.0, max_value=1.0, value=0.5, format="%.4f", key="ht2_p0")
            test_type = st.selectbox("Test Type:", 
                ["Two-tailed (p ≠ p₀)", "Left-tailed (p < p₀)", "Right-tailed (p > p₀)"], key="ht2_type")
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="ht2_alpha")
        
        if st.button("Perform Hypothesis Test", type="primary", key="ht2_btn"):
            try:
                tt = 'two' if 'Two' in test_type else ('left' if 'Left' in test_type else 'right')
                symbols = {'two': '≠', 'left': '<', 'right': '>'}
                
                p_hat = x / n
                se = np.sqrt(p0 * (1 - p0) / n)
                z_stat = (p_hat - p0) / se
                p_value = z_pvalue(z_stat, tt)
                reject = p_value < alpha
                
                cond_met = (n * p0 >= 10) and (n * (1 - p0) >= 10)
                cond_text = '✓ Conditions met' if cond_met else '⚠️ Warning: Conditions not met'
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                conclusion = f"There IS sufficient evidence at α = {alpha} to conclude that p {symbols[tt]} {p0}." if reject else f"There is NOT sufficient evidence at α = {alpha} to conclude that p {symbols[tt]} {p0}."
                
                st.markdown(f"""
                <div class="result-box-ht">
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
                        <strong>Sample Statistics:</strong><br>
                        Successes: x = {x}<br>
                        Sample size: n = {n}<br>
                        Sample proportion: p̂ = {fmt(p_hat)}<br><br>
                        
                        <strong>Conditions Check:</strong> {cond_text}<br>
                        np₀ = {fmt(n*p0, 1)}, n(1-p₀) = {fmt(n*(1-p0), 1)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        Standard Error: SE = √[p₀(1-p₀)/n] = {fmt(se)}<br>
                        Test Statistic: z = (p̂ - p₀)/SE = {fmt(z_stat)}<br>
                        P-value = {fmt(p_value)}<br>
                        α = {alpha}<br><br>
                        
                        <strong>Conclusion:</strong> {conclusion}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- HT: TWO INDEPENDENT MEANS -----
    elif ht_subtab == "Two Independent Means":
        st.markdown('<div class="section-header-ht">Two Sample t-Test (Independent)</div>', unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="ht3_input")
        
        if input_type == "Summary Statistics":
            st.markdown("**Group 1:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                n1 = st.number_input("n₁:", min_value=2, value=30, key="ht3_n1")
            with col2:
                x1 = st.number_input("x̄₁:", value=0.0, format="%.6f", key="ht3_x1")
            with col3:
                s1 = st.number_input("s₁:", min_value=0.0001, value=1.0, format="%.6f", key="ht3_s1")
            
            st.markdown("**Group 2:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                n2 = st.number_input("n₂:", min_value=2, value=30, key="ht3_n2")
            with col2:
                x2 = st.number_input("x̄₂:", value=0.0, format="%.6f", key="ht3_x2")
            with col3:
                s2 = st.number_input("s₂:", min_value=0.0001, value=1.0, format="%.6f", key="ht3_s2")
        else:
            col1, col2 = st.columns(2)
            with col1:
                raw1 = st.text_area("Group 1 Data:", key="ht3_raw1")
            with col2:
                raw2 = st.text_area("Group 2 Data:", key="ht3_raw2")
        
        col1, col2 = st.columns(2)
        with col1:
            test_type = st.selectbox("Test Type:", 
                ["Two-tailed (μ₁ ≠ μ₂)", "Left-tailed (μ₁ < μ₂)", "Right-tailed (μ₁ > μ₂)"], key="ht3_type")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="ht3_alpha")
        
        if st.button("Perform Hypothesis Test", type="primary", key="ht3_btn"):
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
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                conclusion = f"There IS sufficient evidence at α = {alpha} to conclude that μ₁ {symbols[tt]} μ₂." if reject else f"There is NOT sufficient evidence at α = {alpha} to conclude that μ₁ {symbols[tt]} μ₂."
                
                st.markdown(f"""
                <div class="result-box-ht">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> μ₁ = μ₂ (or μ₁ - μ₂ = 0)<br>
                        <strong>Hₐ:</strong> μ₁ {symbols[tt]} μ₂
                    </div>
                    <div class="result-value">
                        t = {fmt(t_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        Group 1: n₁ = {n1}, x̄₁ = {fmt(x1)}, s₁ = {fmt(s1)}<br>
                        Group 2: n₂ = {n2}, x̄₂ = {fmt(x2)}, s₂ = {fmt(s2)}<br>
                        Difference: x̄₁ - x̄₂ = {fmt(diff)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        Standard Error: SE = √[(s₁²/n₁) + (s₂²/n₂)] = {fmt(se)}<br>
                        Degrees of Freedom: df ≈ {fmt(df, 2)} (Welch's)<br>
                        Test Statistic: t = (x̄₁ - x̄₂)/SE = {fmt(t_stat)}<br>
                        P-value = {fmt(p_value)}<br>
                        α = {alpha}<br><br>
                        
                        <strong>Conclusion:</strong> {conclusion}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- HT: PAIRED SAMPLES -----
    elif ht_subtab == "Paired Samples":
        st.markdown('<div class="section-header-ht">Paired t-Test</div>', unsafe_allow_html=True)
        
        input_type = st.radio("Input Type:", ["Summary Statistics", "Raw Data"], horizontal=True, key="ht4_input")
        
        if input_type == "Summary Statistics":
            col1, col2 = st.columns(2)
            with col1:
                n = st.number_input("Number of Pairs (n):", min_value=2, value=30, key="ht4_n")
                d_bar = st.number_input("Mean Difference (d̄):", value=0.0, format="%.6f", key="ht4_dbar")
            with col2:
                s_d = st.number_input("Std Dev of Differences (s_d):", min_value=0.0001, value=1.0, format="%.6f", key="ht4_sd")
        else:
            col1, col2 = st.columns(2)
            with col1:
                before = st.text_area("Before/Group 1:", key="ht4_before")
            with col2:
                after = st.text_area("After/Group 2:", key="ht4_after")
        
        col1, col2 = st.columns(2)
        with col1:
            test_type = st.selectbox("Test Type:", 
                ["Two-tailed (μ_d ≠ 0)", "Left-tailed (μ_d < 0)", "Right-tailed (μ_d > 0)"], key="ht4_type")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="ht4_alpha")
        
        if st.button("Perform Hypothesis Test", type="primary", key="ht4_btn"):
            try:
                if input_type == "Raw Data":
                    before_data = parse_data(before)
                    after_data = parse_data(after)
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
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                conclusion = f"There IS sufficient evidence at α = {alpha} to conclude that μ_d {symbols[tt]} 0." if reject else f"There is NOT sufficient evidence at α = {alpha} to conclude that μ_d {symbols[tt]} 0."
                
                st.markdown(f"""
                <div class="result-box-ht">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> μ_d = 0<br>
                        <strong>Hₐ:</strong> μ_d {symbols[tt]} 0
                    </div>
                    <div class="result-value">
                        t = {fmt(t_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Sample Statistics:</strong><br>
                        Number of pairs: n = {n}<br>
                        Mean difference: d̄ = {fmt(d_bar)}<br>
                        Std dev of differences: s_d = {fmt(s_d)}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        df = n - 1 = {df}<br>
                        SE = s_d/√n = {fmt(se)}<br>
                        t = d̄/SE = {fmt(t_stat)}<br>
                        P-value = {fmt(p_value)}<br>
                        α = {alpha}<br><br>
                        
                        <strong>Conclusion:</strong> {conclusion}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- HT: TWO PROPORTIONS -----
    elif ht_subtab == "Two Proportions":
        st.markdown('<div class="section-header-ht">Two Sample z-Test for Proportions</div>', unsafe_allow_html=True)
        
        st.markdown("**Group 1:**")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("Successes (x₁):", min_value=0, value=50, key="ht5_x1")
        with col2:
            n1 = st.number_input("Sample Size (n₁):", min_value=1, value=100, key="ht5_n1")
        
        st.markdown("**Group 2:**")
        col1, col2 = st.columns(2)
        with col1:
            x2 = st.number_input("Successes (x₂):", min_value=0, value=40, key="ht5_x2")
        with col2:
            n2 = st.number_input("Sample Size (n₂):", min_value=1, value=100, key="ht5_n2")
        
        col1, col2 = st.columns(2)
        with col1:
            test_type = st.selectbox("Test Type:", 
                ["Two-tailed (p₁ ≠ p₂)", "Left-tailed (p₁ < p₂)", "Right-tailed (p₁ > p₂)"], key="ht5_type")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="ht5_alpha")
        
        if st.button("Perform Hypothesis Test", type="primary", key="ht5_btn"):
            try:
                tt = 'two' if 'Two' in test_type else ('left' if 'Left' in test_type else 'right')
                symbols = {'two': '≠', 'left': '<', 'right': '>'}
                
                p1 = x1 / n1
                p2 = x2 / n2
                p_pooled = (x1 + x2) / (n1 + n2)
                
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                z_stat = (p1 - p2) / se
                p_value = z_pvalue(z_stat, tt)
                reject = p_value < alpha
                
                cond_met = all([n1*p_pooled >= 10, n1*(1-p_pooled) >= 10, n2*p_pooled >= 10, n2*(1-p_pooled) >= 10])
                cond_text = '✓ All conditions met' if cond_met else '⚠️ Warning: Some conditions not met'
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                conclusion = f"There IS sufficient evidence at α = {alpha} to conclude that p₁ {symbols[tt]} p₂." if reject else f"There is NOT sufficient evidence at α = {alpha} to conclude that p₁ {symbols[tt]} p₂."
                
                st.markdown(f"""
                <div class="result-box-ht">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> p₁ = p₂<br>
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
                        <strong>Sample Statistics:</strong><br>
                        Group 1: x₁ = {x1}, n₁ = {n1}, p̂₁ = {fmt(p1)}<br>
                        Group 2: x₂ = {x2}, n₂ = {n2}, p̂₂ = {fmt(p2)}<br>
                        Pooled proportion: p̂ = {fmt(p_pooled)}<br><br>
                        
                        <strong>Conditions Check:</strong> {cond_text}<br><br>
                        
                        <strong>Calculations:</strong><br>
                        SE = √[p̂(1-p̂)(1/n₁ + 1/n₂)] = {fmt(se)}<br>
                        z = (p̂₁ - p̂₂)/SE = {fmt(z_stat)}<br>
                        P-value = {fmt(p_value)}<br>
                        α = {alpha}<br><br>
                        
                        <strong>Conclusion:</strong> {conclusion}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================
# TAB 3: ADVANCED TESTS
# ============================================================

with tab_adv:
    adv_subtab = st.selectbox(
        "Select Calculator:",
        ["Chi-Square Goodness of Fit", "Chi-Square Independence", "Chi-Square Homogeneity", 
         "One-Way ANOVA", "Correlation & Regression"],
        key="adv_select"
    )
    
    st.markdown("---")
    
    # ----- CHI-SQUARE GOODNESS OF FIT -----
    if adv_subtab == "Chi-Square Goodness of Fit":
        st.markdown('<div class="section-header-adv">Chi-Square Goodness of Fit Test</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            obs_text = st.text_area("Observed Frequencies:", value="20, 30, 25, 25", 
                                     placeholder="Enter comma-separated values", key="gof_obs")
        with col2:
            exp_text = st.text_area("Expected Frequencies:", value="25, 25, 25, 25",
                                     placeholder="Enter comma-separated values", key="gof_exp")
        
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="gof_alpha")
        
        if st.button("Perform Chi-Square Test", type="primary", key="gof_btn"):
            try:
                obs = parse_data(obs_text)
                exp = parse_data(exp_text)
                
                if len(obs) != len(exp) or len(obs) < 2:
                    st.error("Please enter equal numbers of observed and expected values (at least 2).")
                    st.stop()
                
                chi2_stat = np.sum((obs - exp)**2 / exp)
                df = len(obs) - 1
                p_value = chi2_pvalue(chi2_stat, df)
                reject = p_value < alpha
                
                # Create table
                table_html = "<table style='width:100%; border-collapse:collapse; margin:10px 0;'>"
                table_html += "<tr style='background:#f5f5f5;'><th style='border:1px solid #ddd; padding:10px;'>Category</th><th style='border:1px solid #ddd; padding:10px;'>Observed</th><th style='border:1px solid #ddd; padding:10px;'>Expected</th><th style='border:1px solid #ddd; padding:10px;'>(O-E)²/E</th></tr>"
                for i in range(len(obs)):
                    contrib = (obs[i] - exp[i])**2 / exp[i]
                    table_html += f"<tr><td style='border:1px solid #ddd; padding:10px; text-align:center;'>{i+1}</td><td style='border:1px solid #ddd; padding:10px; text-align:center;'>{fmt(obs[i], 2)}</td><td style='border:1px solid #ddd; padding:10px; text-align:center;'>{fmt(exp[i], 2)}</td><td style='border:1px solid #ddd; padding:10px; text-align:center;'>{fmt(contrib)}</td></tr>"
                table_html += "</table>"
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> The distribution matches the expected frequencies<br>
                        <strong>Hₐ:</strong> The distribution does NOT match the expected frequencies
                    </div>
                    <div class="result-value">
                        χ² = {fmt(chi2_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Data Summary:</strong><br>
                        {table_html}<br>
                        
                        <strong>Calculations:</strong><br>
                        χ² = Σ(O-E)²/E = {fmt(chi2_stat)}<br>
                        df = k - 1 = {len(obs)} - 1 = {df}<br>
                        P-value = {fmt(p_value)}<br>
                        α = {alpha}<br><br>
                        
                        <strong>Conclusion:</strong> {"The distribution does NOT match expected frequencies." if reject else "There is insufficient evidence to conclude the distribution differs from expected."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- CHI-SQUARE INDEPENDENCE -----
    elif adv_subtab == "Chi-Square Independence":
        st.markdown('<div class="section-header-adv">Chi-Square Test of Independence</div>', unsafe_allow_html=True)
        
        matrix_text = st.text_area("Contingency Table:", 
                                    value="30, 20\n25, 25",
                                    placeholder="Enter matrix (rows separated by newlines, values by commas)\nExample:\n30, 20\n25, 25",
                                    height=120, key="indep_matrix")
        
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="indep_alpha")
        
        if st.button("Perform Chi-Square Test", type="primary", key="indep_btn"):
            try:
                rows = matrix_text.strip().split('\n')
                observed = np.array([[float(x) for x in row.split(',')] for row in rows])
                
                row_totals = observed.sum(axis=1)
                col_totals = observed.sum(axis=0)
                grand_total = observed.sum()
                expected = np.outer(row_totals, col_totals) / grand_total
                
                chi2_stat = np.sum((observed - expected)**2 / expected)
                df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
                p_value = chi2_pvalue(chi2_stat, df)
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> The variables are independent<br>
                        <strong>Hₐ:</strong> The variables are NOT independent (associated)
                    </div>
                    <div class="result-value">
                        χ² = {fmt(chi2_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Calculations:</strong><br>
                        χ² = Σ(O-E)²/E = {fmt(chi2_stat)}<br>
                        df = (r-1)(c-1) = ({observed.shape[0]}-1)({observed.shape[1]}-1) = {df}<br>
                        P-value = {fmt(p_value)}<br>
                        α = {alpha}<br><br>
                        
                        <strong>Conclusion:</strong> {"The variables ARE associated (not independent)." if reject else "There is insufficient evidence to conclude the variables are associated."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display tables
                st.markdown("**Observed Frequencies:**")
                st.dataframe(observed, use_container_width=True)
                st.markdown("**Expected Frequencies:**")
                st.dataframe(expected.round(4), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- CHI-SQUARE HOMOGENEITY -----
    elif adv_subtab == "Chi-Square Homogeneity":
        st.markdown('<div class="section-header-adv">Chi-Square Test of Homogeneity</div>', unsafe_allow_html=True)
        
        matrix_text = st.text_area("Contingency Table:", 
                                    value="30, 20, 10\n25, 25, 10",
                                    placeholder="Enter matrix (rows = populations, columns = categories)",
                                    height=120, key="homog_matrix")
        
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="homog_alpha")
        
        if st.button("Perform Chi-Square Test", type="primary", key="homog_btn"):
            try:
                rows = matrix_text.strip().split('\n')
                observed = np.array([[float(x) for x in row.split(',')] for row in rows])
                
                row_totals = observed.sum(axis=1)
                col_totals = observed.sum(axis=0)
                grand_total = observed.sum()
                expected = np.outer(row_totals, col_totals) / grand_total
                
                chi2_stat = np.sum((observed - expected)**2 / expected)
                df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
                p_value = chi2_pvalue(chi2_stat, df)
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> The populations have the same distribution<br>
                        <strong>Hₐ:</strong> The populations have different distributions
                    </div>
                    <div class="result-value">
                        χ² = {fmt(chi2_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Calculations:</strong><br>
                        χ² = {fmt(chi2_stat)}<br>
                        df = (r-1)(c-1) = {df}<br>
                        P-value = {fmt(p_value)}<br>
                        α = {alpha}<br><br>
                        
                        <strong>Conclusion:</strong> {"The populations have DIFFERENT distributions." if reject else "There is insufficient evidence to conclude the populations have different distributions."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- ANOVA -----
    elif adv_subtab == "One-Way ANOVA":
        st.markdown('<div class="section-header-adv">One-Way ANOVA</div>', unsafe_allow_html=True)
        
        groups_text = st.text_area("Groups Data:", 
                                    value="Group 1: 5, 6, 7, 8, 9\nGroup 2: 4, 5, 6, 7, 8\nGroup 3: 3, 4, 5, 6, 7",
                                    placeholder="Enter groups (one per line)\nFormat: Group Name: value1, value2, ...",
                                    height=150, key="anova_groups")
        
        alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="anova_alpha")
        
        if st.button("Perform ANOVA", type="primary", key="anova_btn"):
            try:
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
                    st.stop()
                
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
                df_total = total_n - 1
                
                msb = ssb / df_between
                msw = ssw / df_within
                f_stat = msb / msw
                p_value = f_pvalue(f_stat, df_between, df_within)
                reject = p_value < alpha
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀' if reject else '✓ FAIL TO REJECT H₀'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="hypothesis-box">
                        <strong>H₀:</strong> μ₁ = μ₂ = ... = μₖ (all means are equal)<br>
                        <strong>Hₐ:</strong> At least one mean is different
                    </div>
                    <div class="result-value">
                        F = {fmt(f_stat)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Summary:</strong><br>
                        Grand Mean = {fmt(grand_mean)}<br>
                        Total n = {total_n}<br>
                        Number of groups = {k}<br><br>
                        
                        <strong>Conclusion:</strong> {"At least one group mean is significantly different." if reject else "There is insufficient evidence to conclude any group means differ."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display tables
                st.markdown("**Group Summary:**")
                import pandas as pd
                summary_df = pd.DataFrame({
                    'Group': group_names,
                    'n': group_sizes,
                    'Mean': [round(m, 6) for m in group_means],
                    'Std Dev': [round(s, 6) for s in group_stds]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                st.markdown("**ANOVA Table:**")
                anova_df = pd.DataFrame({
                    'Source': ['Between', 'Within', 'Total'],
                    'SS': [round(ssb, 6), round(ssw, 6), round(sst, 6)],
                    'df': [df_between, df_within, df_total],
                    'MS': [round(msb, 6), round(msw, 6), ''],
                    'F': [round(f_stat, 6), '', '']
                })
                st.dataframe(anova_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ----- CORRELATION & REGRESSION -----
    elif adv_subtab == "Correlation & Regression":
        st.markdown('<div class="section-header-adv">Correlation & Linear Regression</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            x_text = st.text_area("X Data:", value="1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
                                   placeholder="Enter X values", key="corr_x")
        with col2:
            y_text = st.text_area("Y Data:", value="2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.9, 18.1, 20.0",
                                   placeholder="Enter Y values", key="corr_y")
        
        col1, col2 = st.columns(2)
        with col1:
            pred_x = st.number_input("Predict Y for X =", value=5.5, format="%.4f", key="corr_pred")
        with col2:
            alpha = st.selectbox("Significance Level (α):", [0.01, 0.05, 0.10], index=1, key="corr_alpha")
        
        if st.button("Calculate Correlation & Regression", type="primary", key="corr_btn"):
            try:
                x = parse_data(x_text)
                y = parse_data(y_text)
                
                if len(x) != len(y) or len(x) < 3:
                    st.error("Please enter equal numbers of X and Y values (at least 3 pairs).")
                    st.stop()
                
                n = len(x)
                x_mean = np.mean(x)
                y_mean = np.mean(y)
                
                numerator = np.sum((x - x_mean) * (y - y_mean))
                denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
                r = numerator / denominator
                r_squared = r**2
                
                b_slope = numerator / np.sum((x - x_mean)**2)
                a_intercept = y_mean - b_slope * x_mean
                
                t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                df = n - 2
                p_value = t_pvalue(t_stat, df, 'two')
                reject = p_value < alpha
                
                y_pred = a_intercept + b_slope * pred_x
                
                decision_class = 'decision-reject' if reject else 'decision-fail'
                decision_text = '❌ REJECT H₀ (Significant correlation)' if reject else '✓ FAIL TO REJECT H₀ (No significant correlation)'
                
                st.markdown(f"""
                <div class="result-box-ci">
                    <div class="hypothesis-box">
                        <strong>Correlation Test:</strong><br>
                        H₀: ρ = 0 (no linear correlation)<br>
                        Hₐ: ρ ≠ 0 (significant linear correlation)
                    </div>
                    <div class="result-value">
                        r = {fmt(r)}<br>
                        r² = {fmt(r_squared)}<br>
                        P-value = {fmt(p_value)}
                    </div>
                    <div class="{decision_class}">
                        {decision_text}
                    </div>
                    <div class="details-box">
                        <strong>Data Summary:</strong><br>
                        n = {n} pairs<br>
                        x̄ = {fmt(x_mean)}, ȳ = {fmt(y_mean)}<br><br>
                        
                        <strong>Correlation:</strong><br>
                        r = {fmt(r)}<br>
                        r² = {fmt(r_squared)} ({fmt(r_squared*100, 2)}% of variation explained)<br>
                        t = {fmt(t_stat)}, df = {df}<br><br>
                        
                        <strong>Regression Equation:</strong><br>
                        ŷ = {fmt(a_intercept)} + {fmt(b_slope)}x<br>
                        Slope (b) = {fmt(b_slope)}<br>
                        y-intercept (a) = {fmt(a_intercept)}<br><br>
                        
                        <strong>Prediction:</strong><br>
                        When x = {pred_x}: ŷ = <b>{fmt(y_pred)}</b><br><br>
                        
                        <strong>Conclusion:</strong> {"There IS significant linear correlation." if reject else "There is NOT sufficient evidence of linear correlation."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 13px;">
    📊 Statistics Calculator | Built with Streamlit & SciPy | High-Precision Calculations
</div>
""", unsafe_allow_html=True)
