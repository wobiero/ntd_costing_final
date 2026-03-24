"""
Schistosomiasis Endgame Costing Tool
=====================================
Standalone Streamlit application for cost-effectiveness and budget impact
analysis of schistosomiasis (S. mansoni / S. japonicum and S. haematobium)
MDA programmes in sub-Saharan Africa.

Methodological framework
------------------------
- Epidemiological backbone : ESPEN 2020 schistosomiasis dataset
- Disease modules          : intestinal + hepatosplenic (S. mansoni/japonicum)
                             urogenital + bladder cancer + FGS (S. haematobium)
- Uncertainty              : Monte Carlo PSA (n = 1 000), gamma / truncated-normal
                             / log-normal distributions
- Productivity losses      : human capital, inequality-adjusted (bottom quintile wage)
- CEA threshold            : Woods et al. (2016) elasticity approach
- Discounting              : 3 % costs and effects (WHO reference case)
- DALY = YLD + YLL         : YLL included for bladder cancer mortality only
- BIA horizon              : 3 – 5 years, user-selectable

Author  : [Author name]
Version : 1.0.0
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import io
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import altair as alt
import streamlit as st
from PIL import Image

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Schistosomiasis Costing Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Schistosomiasis Endgame Costing Tool v1.0")

# ── Data path ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "datasets"))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – SHARED HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _positive_normal(mu: float, sigma: float) -> float:
    """Draw from N(mu, sigma) truncated to positive values."""
    v = -1.0
    while v <= 0:
        v = random.gauss(mu, sigma)
    return v


def _positive_lognormal(mu: float, sigma: float) -> float:
    """Draw from log-normal(mu, sigma) – mu/sigma are the underlying normal params."""
    v = -1.0
    while v <= 0:
        v = np.random.lognormal(mu, sigma)
    return v


def _gamma(mu: float, sigma: float) -> float:
    """
    Draw from Gamma with given mean (mu) and SD (sigma).
    Shape = mu² / sigma²,  scale = sigma² / mu.
    """
    shape = (mu ** 2) / (sigma ** 2)
    scale = (sigma ** 2) / mu
    v = -1.0
    while v <= 0:
        v = np.random.gamma(shape, scale)
    return v


def _gamma_from_ci(mean: float, ci_lo: float, ci_hi: float, n: int = 1_000) -> np.ndarray:
    """Vectorised Gamma simulation from mean + 95 % CI."""
    se = (ci_hi - ci_lo) / (2 * 1.96)
    shape = mean ** 2 / se ** 2
    scale = se ** 2 / mean
    return np.random.gamma(shape, scale, n)


def _truncated_beta(mu: float, sigma: float) -> float:
    """
    Draw from Beta re-parameterised by mean and SD; clipped to (0, 1).
    Useful for proportions.
    """
    alpha_ = mu * ((mu * (1 - mu)) / sigma ** 2 - 1)
    beta_  = (1 - mu) * ((mu * (1 - mu)) / sigma ** 2 - 1)
    alpha_ = max(alpha_, 0.01)
    beta_  = max(beta_, 0.01)
    return float(np.random.beta(alpha_, beta_))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _format_ci(mean: float, lo: float, hi: float, fmt: str = ",.0f") -> str:
    return f"{mean:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"


def _discount(annual_value: float, horizon: int, rate: float = 0.03) -> float:
    """Present value of a constant annual_value over horizon years."""
    if rate == 0:
        return annual_value * horizon
    return annual_value * (1 - (1 + rate) ** -horizon) / rate


def download_df(df: pd.DataFrame, label: str = "Download CSV") -> None:
    st.download_button(label, df.to_csv(index=False).encode(), file_name="results.csv", mime="text/csv")


def styled_table(df: pd.DataFrame, fmt_cols: Optional[dict] = None) -> None:
    if fmt_cols:
        st.dataframe(df.style.format(fmt_cols))
    else:
        st.dataframe(df)


# ── CEA threshold (Woods et al. 2016 elasticity method) ──────────────────────
@st.cache_data
def cea_threshold(
    gdp_ppp_country: float,
    uk_cet: float   = 26_705,
    gdp_ppp_uk: float = 46_659,
    elasticity: float = 2.478,
) -> float:
    return uk_cet * (gdp_ppp_country / gdp_ppp_uk) ** elasticity


# ── Inequality-adjusted daily wage ────────────────────────────────────────────
def adj_daily_wage(
    annual_ppp: float,
    q1_share: float,   # income share of bottom quintile (0–1)
    weekly_hours: float = 40.0,
    working_days: int   = 300,
) -> float:
    return annual_ppp * q1_share / (0.20 * working_days)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – TERMS & CONDITIONS  →  gate access
# ═══════════════════════════════════════════════════════════════════════════════

_TERMS = """
**By using this tool you agree to the following:**

* Outputs are indicative estimates based on modelled parameters and should not
  replace locally-validated cost data or clinical judgment.
* The authors disclaim liability for programmatic decisions made solely on the
  basis of tool outputs.
* Please cite the accompanying manuscript when using outputs in publications or
  funding proposals.
* All programme cost inputs should be verified against country-specific budgets.
"""

_gate = st.empty()
with _gate.container():
    with st.expander("Terms and conditions"):
        st.markdown(_TERMS)
    _accept = st.selectbox(
        "Accept or decline to proceed",
        ("", "Accept", "Decline"),
        index=0,
    )
    if not _accept:
        st.warning("Please accept or decline the terms and conditions.")
        st.stop()
    if _accept == "Decline":
        st.info("You have declined. Please close the browser tab to exit.")
        st.stop()
_gate.empty()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_country_inputs(path: str) -> pd.DataFrame:
    """
    Load the shared country-level economic parameters file.
    Expected columns (subset): Country, Annual_PPP(Int$), Weekly_Work_Hours,
    inequality_.2_quintile, Life_Expectancy, Inflation rate (consumer prices) (%),
    Primary Hospital_OPD_Costs, Primary Hospital_IPD_Costs, wb_estimate, upper_ppp_cet.
    """
    df = pd.read_csv(path)
    df["Hourly_PPP(Int$)"] = df.apply(
        lambda r: r["Annual_PPP(Int$)"] / (r["Weekly_Work_Hours"] * 52)
        if pd.isna(r.get("Hourly_PPP(Int$)", np.nan)) else r.get("Hourly_PPP(Int$)", np.nan),
        axis=1,
    )
    return df


@st.cache_data
def load_espen_schisto(path: str) -> pd.DataFrame:
    """
    Load ESPEN schistosomiasis dataset.
    Expected columns: ADMIN0, ADMIN1, ADMIN2, IUs_NAME, IU_CODE,
    Species (mansoni | haematobium | both), Prev_SAC, Prev_Adults,
    PopReq, PopTrg, PopTreat, Cov, Sch_MDA_Rounds.
    """
    df = pd.read_csv(path)
    return df


# -- Load datasets (graceful fallback with synthetic demo data) ----------------
try:
    country_inputs = load_country_inputs(str(DATA_DIR / "df_gdp.csv"))
    espen_schisto  = load_espen_schisto(str(DATA_DIR / "schisto_espen.csv"))
    DATA_LOADED = True
except FileNotFoundError:
    st.warning(
        "⚠️ Dataset files not found in `datasets/`. "
        "Running in **demo mode** with synthetic parameters. "
        "Replace `datasets/df_gdp.csv` and `datasets/schisto_espen.csv` for full functionality."
    )
    # Synthetic demo country data
    country_inputs = pd.DataFrame({
        "Country": ["Kenya", "Tanzania", "Uganda", "Ethiopia", "Mozambique"],
        "Annual_PPP(Int$)": [4_890, 2_870, 2_610, 2_440, 1_260],
        "Weekly_Work_Hours": [48, 44, 48, 48, 44],
        "inequality_.2_quintile": [5.7, 6.8, 5.8, 5.2, 5.0],
        "Life_Expectancy": [67, 65, 63, 66, 60],
        "Inflation rate (consumer prices) (%)": [5.5, 4.2, 4.8, 24.0, 7.0],
        "Primary Hospital_OPD_Costs": [7.2, 5.4, 4.9, 3.8, 3.2],
        "Primary Hospital_IPD_Costs": [32.0, 24.0, 21.0, 17.0, 14.5],
        "wb_estimate": [614, 361, 329, 307, 159],
        "upper_ppp_cet": [1_200, 700, 640, 600, 310],
        "Officialfigure": [54e6, 62e6, 47e6, 118e6, 32e6],
        "pop_15_64": ["55%", "55%", "53%", "52%", "52%"],
        "imf_estimate": [1_840, 1_140, 890, 950, 490],
        "Rx": ["PZQ", "PZQ", "PZQ", "PZQ", "PZQ"],
    })
    # Synthetic ESPEN schisto
    espen_schisto = pd.DataFrame({
        "ADMIN0": ["Kenya"] * 4 + ["Tanzania"] * 4,
        "ADMIN1": ["Kisumu", "Kisumu", "Siaya", "Siaya",
                   "Mwanza", "Mwanza", "Shinyanga", "Shinyanga"],
        "ADMIN2": ["Kisumu East", "Kisumu West", "Siaya", "Bondo",
                   "Mwanza City", "Ilemela", "Shinyanga", "Kahama"],
        "IUs_NAME": ["Kisumu East IU", "Kisumu West IU", "Siaya IU", "Bondo IU",
                     "Mwanza City IU", "Ilemela IU", "Shinyanga IU", "Kahama IU"],
        "IU_CODE": [f"IU{i:03d}" for i in range(1, 9)],
        "Species": ["mansoni", "mansoni", "haematobium", "both",
                    "mansoni", "both", "haematobium", "haematobium"],
        "Prev_SAC": [38.2, 45.1, 22.4, 31.0, 52.3, 41.0, 18.7, 25.0],
        "Prev_Adults": [21.0, 28.0, 12.0, 17.0, 34.0, 25.0, 9.0, 13.0],
        "PopReq": [45_000, 62_000, 38_000, 51_000, 88_000, 74_000, 29_000, 41_000],
        "PopTrg": [40_000, 58_000, 35_000, 47_000, 80_000, 68_000, 26_000, 37_000],
        "PopTreat": [32_000, 48_000, 28_000, 39_000, 68_000, 55_000, 21_000, 29_000],
        "Cov": [80.0, 82.7, 80.0, 83.0, 85.0, 80.9, 80.8, 78.4],
        "Sch_MDA_Rounds": [4, 5, 3, 4, 6, 4, 3, 4],
    })
    DATA_LOADED = False

country_dict = country_inputs.set_index("Country").T.to_dict()
country_list = sorted(country_inputs["Country"].tolist())


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – SIDEBAR: COUNTRY + DISEASE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## Schistosomiasis Costing Tool")
st.sidebar.markdown("---")

country = st.sidebar.selectbox("Select country", country_list)

# Filter ESPEN to selected country
espen_country = espen_schisto[espen_schisto["ADMIN0"] == country].copy()

species_options = sorted(espen_country["Species"].unique().tolist())
species_present = ", ".join([s.replace("mansoni", "S. mansoni")
                              .replace("haematobium", "S. haematobium")
                              .replace("both", "both species")
                              for s in species_options])

st.sidebar.info(f"Species in data: **{species_present}**")

disease_choice = st.sidebar.selectbox(
    "Disease module",
    ("S. mansoni / japonicum (intestinal)", "S. haematobium (urogenital)", "Both species"),
)

run_mansoni    = "mansoni" in disease_choice.lower() or "both" in disease_choice.lower()
run_haematobium= "haematobium" in disease_choice.lower() or "both" in disease_choice.lower()

# Admin hierarchy
st.sidebar.markdown("#### Geographical unit")
admin1_list = sorted(espen_country["ADMIN1"].unique().tolist())
admin1_list.insert(0, "National level")
admin1 = st.sidebar.selectbox("Administrative Unit 1", admin1_list)

if admin1 == "National level":
    espen_unit   = espen_country.copy()
    unit_label   = country
    pop_req_mda  = int(espen_unit["PopReq"].sum())
    pop_trt_mda  = int(espen_unit["PopTreat"].sum())
    prev_sac     = float((espen_unit["Prev_SAC"] * espen_unit["PopReq"]).sum() / max(espen_unit["PopReq"].sum(), 1))
    mda_rounds   = int(espen_unit["Sch_MDA_Rounds"].max())
else:
    espen_adm1  = espen_country[espen_country["ADMIN1"] == admin1]
    admin2_list = sorted(espen_adm1["ADMIN2"].unique().tolist())
    admin2_list.insert(0, f"All of {admin1}")
    admin2 = st.sidebar.selectbox("Administrative Unit 2", admin2_list)

    if admin2.startswith("All of"):
        espen_unit  = espen_adm1.copy()
        unit_label  = admin1
    else:
        espen_adm2  = espen_adm1[espen_adm1["ADMIN2"] == admin2]
        iu_list     = sorted(espen_adm2["IUs_NAME"].unique().tolist())
        iu_list.insert(0, f"All of {admin2}")
        iu_sel      = st.sidebar.selectbox("Implementing Unit", iu_list)
        espen_unit  = espen_adm2 if iu_sel.startswith("All of") else espen_adm2[espen_adm2["IUs_NAME"] == iu_sel]
        unit_label  = admin2 if iu_sel.startswith("All of") else iu_sel

    pop_req_mda = int(espen_unit["PopReq"].sum())
    pop_trt_mda = int(espen_unit["PopTreat"].sum())
    prev_sac    = float((espen_unit["Prev_SAC"] * espen_unit["PopReq"]).sum() / max(espen_unit["PopReq"].sum(), 1))
    mda_rounds  = int(espen_unit["Sch_MDA_Rounds"].max())

mda_coverage_pct = round((pop_trt_mda / max(pop_req_mda, 1)) * 100, 1)

# ── Programme parameters sidebar ─────────────────────────────────────────────
prog_exp = st.sidebar.expander("MDA programme parameters")
with prog_exp:
    mda_coverage = st.slider("MDA coverage (%)", 0, 100, int(mda_coverage_pct))
    mda_frequency= st.radio("MDA frequency", ("Annual", "Biennial"), index=0)
    mda_target   = st.radio("MDA target population", ("SAC only", "SAC + at-risk adults"), index=1)
    disc_costs   = st.number_input("Discount rate – costs", value=0.03, step=0.01)
    disc_effects = st.number_input("Discount rate – effects", value=0.03, step=0.01)
    time_horizon = st.slider("Time horizon (years)", 5, 30, 10)
    bia_horizon  = st.slider("Budget impact horizon (years)", 3, 5, 5)

# ── Programme costs sidebar ───────────────────────────────────────────────────
cost_exp = st.sidebar.expander("Programme costs")
with cost_exp:
    pzq_unit_cost   = st.number_input("PZQ tablet cost (USD)", value=0.08, step=0.01)
    pzq_per_person  = st.number_input("Tablets per treatment course", value=6, step=1)
    delivery_cost   = st.number_input("Delivery cost per person treated (USD)", value=0.50, step=0.05)
    mapping_cost    = st.number_input("Mapping / M&E annual cost (USD)", value=5_000.0)
    training_cost   = st.number_input("Training cost per MDA round (USD)", value=3_000.0)
    supervision_cost= st.number_input("Supervision cost per MDA round (USD)", value=2_000.0)
    other_prog_cost = st.number_input("Other annual programme costs (USD)", value=1_000.0)

pzq_drug_cost   = pzq_unit_cost * pzq_per_person * pop_trt_mda
delivery_total  = delivery_cost * pop_trt_mda
annual_prog_cost= pzq_drug_cost + delivery_total + mapping_cost + training_cost + supervision_cost + other_prog_cost

# ── Country economic parameters sidebar ──────────────────────────────────────
econ_exp = st.sidebar.expander("Country economic parameters")
with econ_exp:
    annual_ppp  = st.number_input("Per capita GDP PPP (Int$)",
                                   value=float(country_dict[country]["Annual_PPP(Int$)"]))
    q1_share    = st.number_input("Bottom quintile income share (%)",
                                   value=float(country_dict[country]["inequality_.2_quintile"])) / 100
    weekly_hrs  = st.number_input("Weekly work hours",
                                   value=float(country_dict[country]["Weekly_Work_Hours"]))
    life_exp    = st.number_input("Life expectancy (years)",
                                   value=float(country_dict[country]["Life_Expectancy"]))
    opd_cost_base= st.number_input("OPD unit cost USD (base year)",
                                   value=float(country_dict[country]["Primary Hospital_OPD_Costs"]))
    ipd_cost_base= st.number_input("IPD unit cost USD (base year)",
                                   value=float(country_dict[country]["Primary Hospital_IPD_Costs"]))

daily_wage_adj = adj_daily_wage(annual_ppp, q1_share, weekly_hrs)
inflation      = float(country_dict[country]["Inflation rate (consumer prices) (%)"])
med_inflation  = (inflation + 3) / 100  # medical inflation 3pp above general
opd_cost_curr  = opd_cost_base * (1 + med_inflation) ** 10
ipd_cost_curr  = ipd_cost_base * (1 + med_inflation) ** 10
cet            = cea_threshold(annual_ppp)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – PARAMETER DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

# ── S. mansoni / japonicum parameters ────────────────────────────────────────
@dataclass
class MansonilInputs:
    """
    Epidemiological and economic parameters for intestinal / hepatosplenic
    schistosomiasis (S. mansoni and S. japonicum).

    Disability weights: GBD 2019.
    Clinical proportions: King et al. (2005), van der Werf et al. (2003),
    WHO (2002) Expert Committee.
    Productivity losses: Audibert et al. (1999), Croce et al. (2010).
    """
    n_iterations: int = 1_000

    # ── Infection intensity split (% of infected population) ─────────────────
    # Light infection: 1–99 EPG (S. mansoni) / S. haematobium equivalent
    pct_light: float = 0.60           # 60% light infection
    pct_light_std: float = 0.08

    pct_heavy: float = 0.40           # 40% heavy infection
    pct_heavy_std: float = 0.08

    # ── Morbidity fractions ───────────────────────────────────────────────────
    # Anemia (affects light + heavy, higher in heavy)
    pct_light_anemia: float = 0.18     # 18% of lightly-infected develop anemia
    pct_light_anemia_std: float = 0.05
    pct_heavy_anemia: float = 0.45     # 45% of heavily-infected develop anemia
    pct_heavy_anemia_std: float = 0.08

    # Hepatomegaly (heavy infection only)
    pct_hepatomegaly: float = 0.22     # 22% of heavy infections → hepatomegaly
    pct_hepatomegaly_std: float = 0.06

    # Periportal fibrosis | portal hypertension | varices (cascade from hepatomegaly)
    pct_fibrosis: float = 0.48         # 48% of hepatomegaly → periportal fibrosis
    pct_fibrosis_std: float = 0.10

    pct_portal_htn: float = 0.32       # 32% of fibrosis → portal hypertension
    pct_portal_htn_std: float = 0.10

    pct_varices: float = 0.58          # 58% of portal HTN → esophageal varices
    pct_varices_std: float = 0.12

    # ── Disability weights (GBD 2019) ────────────────────────────────────────
    dw_anemia_mild: float = 0.006
    dw_anemia_mild_std: float = 0.001

    dw_anemia_moderate: float = 0.058
    dw_anemia_moderate_std: float = 0.010

    dw_hepatomegaly: float = 0.021
    dw_hepatomegaly_std: float = 0.006

    dw_fibrosis: float = 0.021         # periportal fibrosis (same as hepatomegaly GBD)
    dw_fibrosis_std: float = 0.006

    dw_ascites: float = 0.222          # portal hypertension with ascites
    dw_ascites_std: float = 0.030

    dw_varices: float = 0.190          # esophageal varices
    dw_varices_std: float = 0.025

    # ── Productivity losses (fraction of working capacity lost) ───────────────
    prod_loss_anemia: float = 0.12
    prod_loss_anemia_std: float = 0.04

    prod_loss_hepatomegaly: float = 0.10
    prod_loss_hepatomegaly_std: float = 0.04

    prod_loss_portal_htn: float = 0.45
    prod_loss_portal_htn_std: float = 0.10

    prod_loss_varices: float = 0.65
    prod_loss_varices_std: float = 0.12

    # ── MDA (praziquantel) efficacy ───────────────────────────────────────────
    cure_rate: float = 0.85            # single-dose cure rate
    cure_rate_std: float = 0.07

    egg_reduction_rate: float = 0.90   # egg reduction rate (proxy for morbidity)
    err_std: float = 0.05

    morbidity_reduction_hepatic: float = 0.70   # hepatic morbidity reduction with MDA
    mrh_std: float = 0.10

    # ── Health system utilisation ─────────────────────────────────────────────
    opd_visits_anemia: float = 2.0     # OPD visits p.a. per anemia case
    opd_visits_anemia_std: float = 0.5

    opd_visits_hepatic: float = 3.5   # OPD visits p.a. per hepatic case
    opd_visits_hepatic_std: float = 0.8

    ipd_days_varices: float = 7.0     # inpatient days per variceal bleed episode
    ipd_days_varices_std: float = 2.0

    pct_varices_bleed_pa: float = 0.08  # % varices cases with bleeding per year
    pvb_std: float = 0.03

    # ── Baseline caseload anchor ──────────────────────────────────────────────
    # Populated at runtime from ESPEN prevalence + at-risk population
    at_risk_pop: float = 0.0
    infected_n: float = 0.0


@dataclass
class HaematobiumInputs:
    """
    Epidemiological and economic parameters for urogenital schistosomiasis
    (S. haematobium), including bladder cancer and female genital schistosomiasis.

    Disability weights: GBD 2019.
    Clinical proportions: WHO (2002), Hollegaard et al., IARC.
    Bladder cancer PAF: IARC Monograph (2012), Botelho et al. (2011).
    """
    n_iterations: int = 1_000

    # ── Morbidity fractions ───────────────────────────────────────────────────
    pct_hematuria: float = 0.62        # 62% of infected → hematuria
    pct_hematuria_std: float = 0.08

    pct_hydronephrosis: float = 0.15   # 15% → obstructive uropathy / hydronephrosis
    pct_hydronephrosis_std: float = 0.05

    pct_fgs: float = 0.75              # 75% of infected females → FGS
    pct_fgs_std: float = 0.08

    # ── Bladder cancer parameters (population-attributable fraction) ──────────
    # PAF = RR–1) / RR * Pe where Pe = prevalence, RR from IARC
    bladder_cancer_rr: float = 4.20    # relative risk of bladder cancer with S. haematobium
    bladder_cancer_rr_std: float = 1.00

    pct_cancer_primary: float = 0.72   # 72% of bladder cancer at primary stage
    pct_cancer_primary_std: float = 0.08

    pct_cancer_metastatic: float = 0.28
    pct_cancer_metastatic_std: float = 0.08

    # Background bladder cancer incidence per 100,000 (country-level default)
    bg_bladder_cancer_rate: float = 3.5
    bg_bladder_cancer_rate_std: float = 0.8

    # Survival (years from diagnosis)
    cancer_survival_primary: float = 5.0
    cancer_survival_meta: float = 1.5

    # ── Disability weights (GBD 2019) ────────────────────────────────────────
    dw_hematuria: float = 0.020
    dw_hematuria_std: float = 0.005

    dw_hydronephrosis: float = 0.149
    dw_hydronephrosis_std: float = 0.020

    dw_fgs: float = 0.048
    dw_fgs_std: float = 0.012

    dw_cancer_primary: float = 0.288
    dw_cancer_primary_std: float = 0.035

    dw_cancer_metastatic: float = 0.540
    dw_cancer_metastatic_std: float = 0.050

    # ── Productivity losses ───────────────────────────────────────────────────
    prod_loss_hematuria: float = 0.08
    prod_loss_hematuria_std: float = 0.03

    prod_loss_hydronephrosis: float = 0.25
    prod_loss_hydronephrosis_std: float = 0.08

    prod_loss_fgs: float = 0.12
    prod_loss_fgs_std: float = 0.04

    prod_loss_cancer: float = 0.75
    prod_loss_cancer_std: float = 0.10

    # ── MDA (praziquantel) efficacy ───────────────────────────────────────────
    cure_rate: float = 0.87
    cure_rate_std: float = 0.06

    egg_reduction_rate: float = 0.91
    err_std: float = 0.04

    morbidity_reduction_urinary: float = 0.75
    mru_std: float = 0.10

    # Bladder cancer not reversed by MDA (established disease)
    cancer_reduction_mda: float = 0.30   # partial reduction in new cancer cases via reduced PAF
    crm_std: float = 0.12

    # ── Health system utilisation ─────────────────────────────────────────────
    opd_visits_hematuria: float = 2.5
    opd_visits_hematuria_std: float = 0.6

    opd_visits_hydronephrosis: float = 4.0
    opd_visits_hydronephrosis_std: float = 1.0

    ipd_days_hydronephrosis: float = 5.0
    ipd_days_hydronephrosis_std: float = 1.5

    opd_visits_cancer: float = 12.0     # oncology visits p.a.
    opd_visits_cancer_std: float = 3.0

    ipd_days_cancer: float = 14.0
    ipd_days_cancer_std: float = 4.0

    # ── Baseline caseload anchor ──────────────────────────────────────────────
    at_risk_pop: float = 0.0
    infected_n: float = 0.0
    female_fraction: float = 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – CASELOAD ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_caseloads_mansoni(
    at_risk_pop: float,
    prev_pct: float,
    m_params: MansonilInputs,
) -> dict:
    """
    Derive clinical caseloads for S. mansoni from at-risk population and prevalence.
    Returns a dict of point estimates used as PSA anchors.
    """
    infected          = at_risk_pop * (prev_pct / 100)
    light_infected    = infected * m_params.pct_light
    heavy_infected    = infected * m_params.pct_heavy

    anemia_cases      = (light_infected * m_params.pct_light_anemia
                         + heavy_infected * m_params.pct_heavy_anemia)
    hepatomegaly_cases= heavy_infected * m_params.pct_hepatomegaly
    fibrosis_cases    = hepatomegaly_cases * m_params.pct_fibrosis
    portal_htn_cases  = fibrosis_cases * m_params.pct_portal_htn
    varices_cases     = portal_htn_cases * m_params.pct_varices

    return {
        "infected":           infected,
        "light_infected":     light_infected,
        "heavy_infected":     heavy_infected,
        "anemia":             anemia_cases,
        "hepatomegaly":       hepatomegaly_cases,
        "fibrosis":           fibrosis_cases,
        "portal_htn":         portal_htn_cases,
        "varices":            varices_cases,
    }


def estimate_caseloads_haematobium(
    at_risk_pop: float,
    prev_pct: float,
    h_params: HaematobiumInputs,
    female_fraction: float = 0.50,
) -> dict:
    """
    Derive clinical caseloads for S. haematobium.
    Bladder cancer PAF = (RR – 1) / RR * Pe.
    """
    infected            = at_risk_pop * (prev_pct / 100)
    pe                  = prev_pct / 100  # prevalence in at-risk pop

    hematuria_cases     = infected * h_params.pct_hematuria
    hydronephrosis_cases= infected * h_params.pct_hydronephrosis
    fgs_cases           = infected * female_fraction * h_params.pct_fgs

    # Bladder cancer PAF
    rr = h_params.bladder_cancer_rr
    paf = (rr - 1) / rr * pe
    bg_cancer_per_person = h_params.bg_bladder_cancer_rate / 100_000
    total_bladder_cancer = at_risk_pop * bg_cancer_per_person * (1 + paf * (rr - 1))
    cancer_primary       = total_bladder_cancer * h_params.pct_cancer_primary
    cancer_metastatic    = total_bladder_cancer * h_params.pct_cancer_metastatic

    return {
        "infected":            infected,
        "hematuria":           hematuria_cases,
        "hydronephrosis":      hydronephrosis_cases,
        "fgs":                 fgs_cases,
        "paf":                 paf,
        "bladder_cancer_total":total_bladder_cancer,
        "cancer_primary":      cancer_primary,
        "cancer_metastatic":   cancer_metastatic,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – MONTE CARLO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _single_run_mansoni(m: MansonilInputs, caseloads: dict) -> dict:
    """One Monte Carlo iteration for S. mansoni module."""
    # Draw stochastic parameters
    pl      = _truncated_beta(m.pct_light, m.pct_light_std)
    ph      = 1 - pl
    pla     = _truncated_beta(m.pct_light_anemia, m.pct_light_anemia_std)
    pha     = _truncated_beta(m.pct_heavy_anemia, m.pct_heavy_anemia_std)
    phep    = _truncated_beta(m.pct_hepatomegaly, m.pct_hepatomegaly_std)
    pfib    = _truncated_beta(m.pct_fibrosis, m.pct_fibrosis_std)
    ppht    = _truncated_beta(m.pct_portal_htn, m.pct_portal_htn_std)
    pvar    = _truncated_beta(m.pct_varices, m.pct_varices_std)

    dw_am   = _positive_normal(m.dw_anemia_mild, m.dw_anemia_mild_std)
    dw_amod = _positive_normal(m.dw_anemia_moderate, m.dw_anemia_moderate_std)
    dw_hep  = _positive_normal(m.dw_hepatomegaly, m.dw_hepatomegaly_std)
    dw_fib  = _positive_normal(m.dw_fibrosis, m.dw_fibrosis_std)
    dw_asc  = _positive_normal(m.dw_ascites, m.dw_ascites_std)
    dw_var  = _positive_normal(m.dw_varices, m.dw_varices_std)

    pl_prod = _positive_normal(m.prod_loss_anemia, m.prod_loss_anemia_std)
    ph_prod = _positive_normal(m.prod_loss_hepatomegaly, m.prod_loss_hepatomegaly_std)
    pph_prod= _positive_normal(m.prod_loss_portal_htn, m.prod_loss_portal_htn_std)
    pv_prod = _positive_normal(m.prod_loss_varices, m.prod_loss_varices_std)

    cr      = _truncated_beta(m.cure_rate, m.cure_rate_std)
    err     = _truncated_beta(m.egg_reduction_rate, m.err_std)
    mrh     = _truncated_beta(m.morbidity_reduction_hepatic, m.mrh_std)

    ov_an   = _positive_normal(m.opd_visits_anemia, m.opd_visits_anemia_std)
    ov_hep  = _positive_normal(m.opd_visits_hepatic, m.opd_visits_hepatic_std)
    id_var  = _positive_normal(m.ipd_days_varices, m.ipd_days_varices_std)
    pvb     = _truncated_beta(m.pct_varices_bleed_pa, m.pvb_std)

    # ── Derive caseloads from stochastic proportions ──────────────────────────
    infected    = caseloads["infected"]
    light_inf   = infected * pl
    heavy_inf   = infected * ph
    anemia      = light_inf * pla + heavy_inf * pha
    hepatomeg   = heavy_inf * phep
    fibrosis    = hepatomeg * pfib
    portal_htn  = fibrosis * ppht
    varices     = portal_htn * pvar

    # ── DALYs (YLD only – schisto mortality negligible except cancer) ─────────
    # Split anemia: 60% mild, 40% moderate for light; 30/70 for heavy
    anemia_mild = light_inf * pla * 0.60 + heavy_inf * pha * 0.30
    anemia_mod  = anemia - anemia_mild

    daly_anemia    = anemia_mild * dw_am + anemia_mod * dw_amod
    daly_hepatomeg = hepatomeg * dw_hep
    daly_fibrosis  = fibrosis * dw_fib
    daly_portal    = portal_htn * dw_asc
    daly_varices   = varices * dw_var
    daly_total     = daly_anemia + daly_hepatomeg + daly_fibrosis + daly_portal + daly_varices

    # MDA scenario DALYs
    daly_anemia_mda    = daly_anemia    * (1 - err)
    daly_hepatomeg_mda = daly_hepatomeg * (1 - mrh)
    daly_fibrosis_mda  = daly_fibrosis  * (1 - mrh)
    daly_portal_mda    = daly_portal    * (1 - mrh)
    daly_varices_mda   = daly_varices   * (1 - mrh)
    daly_total_mda     = (daly_anemia_mda + daly_hepatomeg_mda + daly_fibrosis_mda
                          + daly_portal_mda + daly_varices_mda)

    # ── Productivity losses ───────────────────────────────────────────────────
    work_days_lost = (
        anemia * pl_prod * 300
        + hepatomeg * ph_prod * 300
        + portal_htn * pph_prod * 300
        + varices * pv_prod * 300
    )
    work_days_lost_mda = work_days_lost * (1 - mrh)

    # ── Health sector utilisation ────────────────────────────────────────────
    opd_anemia   = anemia * ov_an
    opd_hepatic  = (hepatomeg + fibrosis + portal_htn) * ov_hep
    ipd_varices  = varices * pvb * id_var      # only bleeding episodes admitted
    opd_total    = opd_anemia + opd_hepatic
    opd_total_mda= opd_total * (1 - mrh)
    ipd_total    = ipd_varices
    ipd_total_mda= ipd_varices * (1 - mrh)

    return dict(
        infected=infected, anemia=anemia, hepatomeg=hepatomeg,
        fibrosis=fibrosis, portal_htn=portal_htn, varices=varices,
        daly_anemia=daly_anemia, daly_hepatomeg=daly_hepatomeg,
        daly_fibrosis=daly_fibrosis, daly_portal=daly_portal,
        daly_varices=daly_varices, daly_total=daly_total,
        daly_anemia_mda=daly_anemia_mda, daly_hepatomeg_mda=daly_hepatomeg_mda,
        daly_fibrosis_mda=daly_fibrosis_mda, daly_portal_mda=daly_portal_mda,
        daly_varices_mda=daly_varices_mda, daly_total_mda=daly_total_mda,
        work_days_lost=work_days_lost, work_days_lost_mda=work_days_lost_mda,
        opd_total=opd_total, opd_total_mda=opd_total_mda,
        ipd_total=ipd_total, ipd_total_mda=ipd_total_mda,
        cure_rate=cr, egg_reduction=err, morbidity_reduction=mrh,
    )


def _single_run_haematobium(h: HaematobiumInputs, caseloads: dict, life_exp: float) -> dict:
    """One Monte Carlo iteration for S. haematobium module."""
    # Stochastic draws
    ph_hem  = _truncated_beta(h.pct_hematuria, h.pct_hematuria_std)
    ph_hyd  = _truncated_beta(h.pct_hydronephrosis, h.pct_hydronephrosis_std)
    ph_fgs  = _truncated_beta(h.pct_fgs, h.pct_fgs_std)
    rr      = _gamma(h.bladder_cancer_rr, h.bladder_cancer_rr_std)
    bg_ca   = _positive_normal(h.bg_bladder_cancer_rate, h.bg_bladder_cancer_rate_std)
    pcp     = _truncated_beta(h.pct_cancer_primary, h.pct_cancer_primary_std)

    dw_hem  = _positive_normal(h.dw_hematuria, h.dw_hematuria_std)
    dw_hyd  = _positive_normal(h.dw_hydronephrosis, h.dw_hydronephrosis_std)
    dw_fgs  = _positive_normal(h.dw_fgs, h.dw_fgs_std)
    dw_cap  = _positive_normal(h.dw_cancer_primary, h.dw_cancer_primary_std)
    dw_cam  = _positive_normal(h.dw_cancer_metastatic, h.dw_cancer_metastatic_std)

    pl_hem  = _positive_normal(h.prod_loss_hematuria, h.prod_loss_hematuria_std)
    pl_hyd  = _positive_normal(h.prod_loss_hydronephrosis, h.prod_loss_hydronephrosis_std)
    pl_fgs  = _positive_normal(h.prod_loss_fgs, h.prod_loss_fgs_std)
    pl_ca   = _positive_normal(h.prod_loss_cancer, h.prod_loss_cancer_std)

    cr      = _truncated_beta(h.cure_rate, h.cure_rate_std)
    err     = _truncated_beta(h.egg_reduction_rate, h.err_std)
    mru     = _truncated_beta(h.morbidity_reduction_urinary, h.mru_std)
    crm     = _truncated_beta(h.cancer_reduction_mda, h.crm_std)

    ov_hem  = _positive_normal(h.opd_visits_hematuria, h.opd_visits_hematuria_std)
    ov_hyd  = _positive_normal(h.opd_visits_hydronephrosis, h.opd_visits_hydronephrosis_std)
    id_hyd  = _positive_normal(h.ipd_days_hydronephrosis, h.ipd_days_hydronephrosis_std)
    ov_ca   = _positive_normal(h.opd_visits_cancer, h.opd_visits_cancer_std)
    id_ca   = _positive_normal(h.ipd_days_cancer, h.ipd_days_cancer_std)

    # Caseloads
    infected   = caseloads["infected"]
    pe         = infected / max(h.at_risk_pop, 1)
    hematuria  = infected * ph_hem
    hydronepr  = infected * ph_hyd
    fgs        = infected * h.female_fraction * ph_fgs

    paf        = (rr - 1) / rr * pe
    bg_per_p   = bg_ca / 100_000
    total_ca   = h.at_risk_pop * bg_per_p * (1 + paf * (rr - 1))
    ca_primary = total_ca * pcp
    ca_meta    = total_ca * (1 - pcp)

    # ── DALYs ────────────────────────────────────────────────────────────────
    # YLD
    daly_hem   = hematuria * dw_hem
    daly_hyd   = hydronepr * dw_hyd
    daly_fgs   = fgs * dw_fgs
    daly_cap   = ca_primary * dw_cap * h.cancer_survival_primary
    daly_cam   = ca_meta * dw_cam * h.cancer_survival_meta

    # YLL (bladder cancer) – simplified: YLL = N_deaths × remaining LE at mean age 55
    mean_age_cancer   = 55.0
    yll_primary = ca_primary * 0.35 * max(life_exp - mean_age_cancer, 0)  # 35% die within 5 yr
    yll_meta    = ca_meta * 0.85 * max(life_exp - mean_age_cancer, 0)     # 85% die within 2 yr

    daly_cancer= daly_cap + daly_cam + yll_primary + yll_meta
    daly_total = daly_hem + daly_hyd + daly_fgs + daly_cancer

    # MDA scenario
    daly_hem_mda   = daly_hem * (1 - mru)
    daly_hyd_mda   = daly_hyd * (1 - mru)
    daly_fgs_mda   = daly_fgs * (1 - mru)
    daly_cancer_mda= daly_cancer * (1 - crm)
    daly_total_mda = daly_hem_mda + daly_hyd_mda + daly_fgs_mda + daly_cancer_mda

    # ── Productivity ─────────────────────────────────────────────────────────
    work_days_lost = (
        hematuria * pl_hem * 300
        + hydronepr * pl_hyd * 300
        + fgs * pl_fgs * 300
        + total_ca * pl_ca * 300
    )
    work_days_lost_mda = (
        hematuria * (1 - mru) * pl_hem * 300
        + hydronepr * (1 - mru) * pl_hyd * 300
        + fgs * (1 - mru) * pl_fgs * 300
        + total_ca * (1 - crm) * pl_ca * 300
    )

    # ── Health sector ────────────────────────────────────────────────────────
    opd_total    = hematuria * ov_hem + hydronepr * ov_hyd + total_ca * ov_ca
    ipd_total    = hydronepr * id_hyd + total_ca * id_ca
    opd_total_mda= (hematuria * (1-mru) * ov_hem + hydronepr * (1-mru) * ov_hyd
                    + total_ca * (1-crm) * ov_ca)
    ipd_total_mda= hydronepr * (1-mru) * id_hyd + total_ca * (1-crm) * id_ca

    return dict(
        infected=infected, hematuria=hematuria, hydronephrosis=hydronepr,
        fgs=fgs, ca_primary=ca_primary, ca_meta=ca_meta, total_ca=total_ca,
        paf=paf,
        daly_hem=daly_hem, daly_hyd=daly_hyd, daly_fgs=daly_fgs,
        daly_cancer=daly_cancer, daly_total=daly_total,
        daly_hem_mda=daly_hem_mda, daly_hyd_mda=daly_hyd_mda,
        daly_fgs_mda=daly_fgs_mda, daly_cancer_mda=daly_cancer_mda,
        daly_total_mda=daly_total_mda,
        work_days_lost=work_days_lost, work_days_lost_mda=work_days_lost_mda,
        opd_total=opd_total, opd_total_mda=opd_total_mda,
        ipd_total=ipd_total, ipd_total_mda=ipd_total_mda,
        cure_rate=cr, egg_reduction=err, morbidity_reduction=mru, cancer_reduction=crm,
        yll_primary=yll_primary, yll_meta=yll_meta,
    )


@st.cache_data
def run_monte_carlo_mansoni(
    n: int,
    at_risk_pop: float,
    prev_pct: float,
    _params_key: str,     # cache-buster – pass str(m_params) or a hash
) -> pd.DataFrame:
    """
    Run n Monte Carlo iterations for S. mansoni module.
    _params_key is used only for cache invalidation; not accessed in the function body.
    """
    m = MansonilInputs(n_iterations=n, at_risk_pop=at_risk_pop)
    cl = estimate_caseloads_mansoni(at_risk_pop, prev_pct, m)
    rows = [_single_run_mansoni(m, cl) for _ in range(n)]
    return pd.DataFrame(rows)


@st.cache_data
def run_monte_carlo_haematobium(
    n: int,
    at_risk_pop: float,
    prev_pct: float,
    female_fraction: float,
    life_exp: float,
    _params_key: str,
) -> pd.DataFrame:
    h = HaematobiumInputs(
        n_iterations=n,
        at_risk_pop=at_risk_pop,
        female_fraction=female_fraction,
    )
    cl = estimate_caseloads_haematobium(at_risk_pop, prev_pct, h, female_fraction)
    rows = [_single_run_haematobium(h, cl, life_exp) for _ in range(n)]
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – ECONOMIC SUMMARY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def daly_summary_table(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Build a formatted DALY summary with 90% uncertainty intervals."""
    if species == "mansoni":
        cols = {
            "Anemia": ("daly_anemia", "daly_anemia_mda"),
            "Hepatomegaly": ("daly_hepatomeg", "daly_hepatomeg_mda"),
            "Periportal fibrosis": ("daly_fibrosis", "daly_fibrosis_mda"),
            "Portal hypertension": ("daly_portal", "daly_portal_mda"),
            "Esophageal varices": ("daly_varices", "daly_varices_mda"),
            "Total": ("daly_total", "daly_total_mda"),
        }
    else:
        cols = {
            "Hematuria": ("daly_hem", "daly_hem_mda"),
            "Hydronephrosis": ("daly_hyd", "daly_hyd_mda"),
            "Female genital schistosomiasis": ("daly_fgs", "daly_fgs_mda"),
            "Bladder cancer (YLD + YLL)": ("daly_cancer", "daly_cancer_mda"),
            "Total": ("daly_total", "daly_total_mda"),
        }

    rows = []
    for label, (c_no, c_mda) in cols.items():
        if c_no not in df.columns:
            continue
        rows.append({
            "Outcome": label,
            "No-MDA mean": df[c_no].mean(),
            "No-MDA 5th": df[c_no].quantile(0.05),
            "No-MDA 95th": df[c_no].quantile(0.95),
            "MDA mean": df[c_mda].mean(),
            "MDA 5th": df[c_mda].quantile(0.05),
            "MDA 95th": df[c_mda].quantile(0.95),
            "DALYs averted": df[c_no].mean() - df[c_mda].mean(),
        })
    return pd.DataFrame(rows)


def productivity_summary(df: pd.DataFrame, daily_wage: float) -> dict:
    mean_no  = df["work_days_lost"].mean()
    lo_no    = df["work_days_lost"].quantile(0.05)
    hi_no    = df["work_days_lost"].quantile(0.95)
    mean_mda = df["work_days_lost_mda"].mean()
    lo_mda   = df["work_days_lost_mda"].quantile(0.05)
    hi_mda   = df["work_days_lost_mda"].quantile(0.95)
    days_gained = mean_no - mean_mda
    econ_gain   = days_gained * daily_wage
    return dict(
        mean_no=mean_no, lo_no=lo_no, hi_no=hi_no,
        mean_mda=mean_mda, lo_mda=lo_mda, hi_mda=hi_mda,
        days_gained=days_gained,
        econ_gain_pa=econ_gain,
        econ_gain_disc=_discount(econ_gain, time_horizon, disc_effects),
    )


def health_sector_costs(df: pd.DataFrame, opd_c: float, ipd_c: float) -> dict:
    hs_no  = df["opd_total"].mean() * opd_c + df["ipd_total"].mean() * ipd_c
    hs_mda = df["opd_total_mda"].mean() * opd_c + df["ipd_total_mda"].mean() * ipd_c
    return dict(
        hs_cost_no=hs_no,
        hs_cost_mda=hs_mda,
        hs_savings_pa=hs_no - hs_mda,
        hs_savings_disc=_discount(hs_no - hs_mda, time_horizon, disc_costs),
    )


def compute_icer(df: pd.DataFrame, annual_prog_cost: float) -> dict:
    """
    ICER = incremental programme cost / DALYs averted per year.
    Also computes NMB and CEAC probabilities.
    """
    dalys_averted = df["daly_total"].values - df["daly_total_mda"].values
    icer_vec = np.where(dalys_averted > 0, annual_prog_cost / dalys_averted, np.nan)

    wtp_range = np.linspace(0, annual_ppp * 5, 500)
    nmb_vec   = dalys_averted * wtp_range[:, None] - annual_prog_cost   # shape (500, n)
    ceac_probs= (nmb_vec > 0).mean(axis=1)

    icer_mean = np.nanmean(icer_vec)
    icer_lo   = np.nanpercentile(icer_vec, 5)
    icer_hi   = np.nanpercentile(icer_vec, 95)

    return dict(
        dalys_averted_mean=dalys_averted.mean(),
        dalys_averted_lo=np.percentile(dalys_averted, 5),
        dalys_averted_hi=np.percentile(dalys_averted, 95),
        icer_mean=icer_mean, icer_lo=icer_lo, icer_hi=icer_hi,
        ceac_wtp=wtp_range,
        ceac_prob=ceac_probs,
        pct_cost_effective_cet=float((icer_vec < cet).mean()) if cet > 0 else np.nan,
    )


def compute_roi(
    hs_savings: float,
    econ_gain: float,
    prog_cost: float,
) -> float:
    """Return on investment: economic benefits per $ invested in MDA."""
    return (hs_savings + econ_gain) / max(prog_cost, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – BUDGET IMPACT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def budget_impact_analysis(
    annual_prog_cost: float,
    hs_savings_pa: float,
    econ_gain_pa: float,
    horizon: int,
    disc_rate: float,
    pzq_cost: float,
    pop_treat: float,
    pzq_per_person: float,
    delivery_c: float,
    freq: str = "Annual",
) -> pd.DataFrame:
    """
    Project discounted costs and benefits over `horizon` years.
    Biennial MDA means every other year has no programme cost.
    Returns a year-by-year DataFrame.
    """
    rows = []
    for yr in range(1, horizon + 1):
        disc = (1 + disc_rate) ** yr
        mda_this_year = (yr % 2 == 1) if freq == "Biennial" else True

        if mda_this_year:
            drug_cost  = pzq_cost * pzq_per_person * pop_treat / disc
            deliv_cost = delivery_c * pop_treat / disc
            other_cost = (annual_prog_cost - pzq_cost * pzq_per_person * pop_treat
                          - delivery_c * pop_treat) / disc
            prog_cost_yr = drug_cost + deliv_cost + other_cost
        else:
            drug_cost = deliv_cost = prog_cost_yr = 0.0
            other_cost = (annual_prog_cost * 0.20) / disc   # 20% fixed costs even in off years

        hs_sav_disc  = hs_savings_pa / disc
        econ_disc    = econ_gain_pa / disc
        net_benefit  = hs_sav_disc + econ_disc - prog_cost_yr

        rows.append(dict(
            Year=2024 + yr,
            MDA_delivered=mda_this_year,
            Drug_costs_USD=drug_cost,
            Delivery_costs_USD=deliv_cost,
            Other_prog_USD=other_cost,
            Total_prog_cost_USD=prog_cost_yr,
            Health_sector_savings_USD=hs_sav_disc,
            Economic_gains_USD=econ_disc,
            Net_benefit_USD=net_benefit,
            Cumulative_net_benefit_USD=0.0,  # filled below
        ))

    df = pd.DataFrame(rows)
    df["Cumulative_net_benefit_USD"] = df["Net_benefit_USD"].cumsum()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ceac(ceac_wtp: np.ndarray, ceac_prob: np.ndarray, cet: float) -> alt.Chart:
    df = pd.DataFrame({"WTP (USD/DALY)": ceac_wtp, "Probability cost-effective": ceac_prob})
    base = alt.Chart(df).mark_line(color="#1A5276").encode(
        x=alt.X("WTP (USD/DALY):Q", title="Willingness-to-pay threshold (USD per DALY averted)"),
        y=alt.Y("Probability cost-effective:Q", axis=alt.Axis(format="%"), title="Probability cost-effective"),
        tooltip=["WTP (USD/DALY)", alt.Tooltip("Probability cost-effective", format=".1%")],
    ).properties(title="Cost-effectiveness acceptability curve (CEAC)", height=300)

    threshold_line = (
        alt.Chart(pd.DataFrame({"WTP": [cet]}))
        .mark_rule(color="#E74C3C", strokeDash=[6, 3])
        .encode(x="WTP:Q")
    )
    return (base + threshold_line).interactive()


def plot_bia(bia_df: pd.DataFrame) -> alt.Chart:
    df_long = bia_df[["Year", "Total_prog_cost_USD", "Health_sector_savings_USD",
                       "Economic_gains_USD"]].melt("Year", var_name="Component", value_name="USD")
    label_map = {
        "Total_prog_cost_USD": "Programme cost",
        "Health_sector_savings_USD": "Health sector savings",
        "Economic_gains_USD": "Productivity gains",
    }
    df_long["Component"] = df_long["Component"].map(label_map)
    colour_scale = alt.Scale(
        domain=list(label_map.values()),
        range=["#C0392B", "#1A7851", "#1A5276"],
    )
    bars = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("USD:Q", title="USD (discounted)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Component:N", scale=colour_scale),
            tooltip=["Year", "Component", alt.Tooltip("USD:Q", format="$,.0f")],
        )
    )
    cum_line = (
        alt.Chart(bia_df)
        .mark_line(point=True, color="#F39C12", strokeWidth=2)
        .encode(
            x=alt.X("Year:O"),
            y=alt.Y("Cumulative_net_benefit_USD:Q", title="Cumulative net benefit (USD)",
                    axis=alt.Axis(format="$,.0f")),
            tooltip=["Year", alt.Tooltip("Cumulative_net_benefit_USD:Q", format="$,.0f")],
        )
    )
    return (
        alt.layer(bars, cum_line)
        .resolve_scale(y="independent")
        .properties(title="Budget impact analysis – discounted costs and benefits", height=320)
        .interactive()
    )


def plot_daly_breakdown(daly_df: pd.DataFrame) -> alt.Chart:
    sub = daly_df[daly_df["Outcome"] != "Total"].copy()
    df_long = sub.melt(
        id_vars=["Outcome"],
        value_vars=["No-MDA mean", "MDA mean"],
        var_name="Scenario",
        value_name="DALYs",
    )
    return (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("DALYs:Q", title="DALYs per year", axis=alt.Axis(format=",.0f")),
            y=alt.Y("Outcome:N", sort="-x"),
            color=alt.Color("Scenario:N", scale=alt.Scale(
                domain=["No-MDA mean", "MDA mean"], range=["#C0392B", "#1A7851"]
            )),
            tooltip=["Outcome", "Scenario", alt.Tooltip("DALYs:Q", format=",.0f")],
        )
        .properties(title="Annual DALYs by outcome: no-MDA vs MDA scenario", height=280)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 – STREAMLIT UI  (tabs)
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📋 About",
    "🌍 Country inputs",
    "🦠 Disease inputs",
    "📊 Results",
    "💰 Budget impact",
    "⚙️ Technical assumptions",
    "📬 Contact",
])

# ── TAB 0: About ──────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("""
    ### Welcome to the Schistosomiasis Endgame Costing Tool

    This tool supports national NTD programme managers, health economists, and
    policymakers in generating economic evidence for schistosomiasis MDA programmes
    across sub-Saharan Africa.

    #### What the tool does
    - Estimates the disease burden of *S. mansoni / japonicum* (intestinal and
      hepatosplenic disease) and *S. haematobium* (urogenital disease, female genital
      schistosomiasis, bladder cancer) using ESPEN surveillance data.
    - Runs **1 000-iteration Monte Carlo probabilistic sensitivity analysis** to
      quantify uncertainty across all economic outputs.
    - Computes **DALYs, ICERs, productivity losses, health sector costs, ROI**, and
      a **budget impact analysis** at national, sub-national, or implementing-unit level.
    - Uses an **inequality-adjusted human capital** approach for productivity losses
      and the **Woods et al. (2016)** elasticity method for the CEA threshold.

    #### How to use it
    1. Select your **country** and **administrative unit** in the sidebar.
    2. Choose the **disease module** (S. mansoni, S. haematobium, or both).
    3. Review and adjust **programme parameters** and **economic defaults** in the sidebar.
    4. Navigate to the **Results** and **Budget Impact** tabs to view outputs.
    5. Download raw Monte Carlo data for further analysis.

    #### Data sources
    - Epidemiology: ESPEN 2020 schistosomiasis dataset
    - Economics: World Bank WDI, WHO-CHOICE, CIA World Factbook, ILO
    - Disability weights: GBD 2019
    - Clinical parameters: WHO (2002), King et al. (2005), van der Werf et al. (2003),
      IARC Monograph (2012)
    """)

    with st.expander("Notes for users"):
        st.markdown("""
        - Default programme costs are **placeholders**. Replace them with locally
          validated figures before using outputs in funding proposals.
        - The bladder cancer model uses a population-attributable fraction (PAF)
          approach based on IARC relative risk estimates and background cancer incidence.
        - MDA does not reverse established hepatosplenic or bladder cancer disease;
          morbidity reduction applies to new disease accumulation only.
        - All monetary outputs are in USD at approximate 2023 values.
        """)

# ── TAB 1: Country inputs ─────────────────────────────────────────────────────
with tabs[1]:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader(f"Selected unit: {unit_label}")
        st.metric("Population requiring MDA", f"{pop_req_mda:,.0f}")
        st.metric("Population treated (last round)", f"{pop_trt_mda:,.0f}")
        st.metric("MDA coverage", f"{mda_coverage_pct:.1f}%")
        st.metric("Cumulative MDA rounds", str(mda_rounds))
        st.metric("SAC prevalence (%)", f"{prev_sac:.1f}%")

    with col2:
        st.subheader("Economic parameters")
        econ_display = pd.DataFrame({
            "Parameter": [
                "Per capita GDP PPP (Int$)",
                "Bottom quintile income share",
                "Inequality-adjusted daily wage (USD)",
                "Weekly work hours",
                "Life expectancy (years)",
                "General inflation rate (%)",
                "Medical cost inflation (approx.)",
                "OPD unit cost – current USD",
                "IPD unit cost – current USD",
                "CEA threshold – Woods et al. (USD/DALY)",
                "CEA threshold – GDP-based 1× (USD/DALY)",
            ],
            "Value": [
                f"{annual_ppp:,.0f}",
                f"{q1_share * 100:.1f}%",
                f"${daily_wage_adj:.2f}",
                f"{weekly_hrs:.0f}",
                f"{life_exp:.0f}",
                f"{inflation:.1f}%",
                f"{med_inflation * 100:.1f}%",
                f"${opd_cost_curr:.2f}",
                f"${ipd_cost_curr:.2f}",
                f"${cet:,.0f}",
                f"${annual_ppp:,.0f}",
            ],
        })
        st.table(econ_display)

    st.markdown("---")
    with st.expander("ESPEN unit-level data"):
        st.dataframe(espen_unit.reset_index(drop=True))

    with st.expander("Annual programme cost breakdown"):
        prog_cost_display = pd.DataFrame({
            "Cost category": [
                "PZQ drug cost",
                "Delivery cost",
                "Mapping / M&E",
                "Training",
                "Supervision",
                "Other",
                "TOTAL",
            ],
            "Annual USD": [
                pzq_drug_cost, delivery_total,
                mapping_cost, training_cost,
                supervision_cost, other_prog_cost,
                annual_prog_cost,
            ],
        })
        st.table(prog_cost_display.style.format({"Annual USD": "${:,.0f}"}))

# ── TAB 2: Disease inputs ─────────────────────────────────────────────────────
with tabs[2]:
    if run_mansoni:
        st.subheader("S. mansoni / japonicum — disease inputs")
        col1, col2 = st.columns(2)
        with col1:
            m_prev = st.number_input(
                "S. mansoni SAC prevalence (%)",
                value=round(prev_sac, 1), min_value=0.0, max_value=100.0,
            )
            m_pop = st.number_input(
                "At-risk population (S. mansoni)",
                value=float(pop_req_mda),
            )
            m_heavy_pct = st.slider("% heavy infection (≥400 EPG)", 10, 70, 40)
        with col2:
            m_hepatomeg_pct = st.slider("% heavy infection → hepatomegaly", 5, 50, 22)
            m_morbidity_red = st.slider(
                "Hepatic morbidity reduction with MDA (%)", 30, 90, 70
            )
            m_cure = st.slider("PZQ cure rate (%)", 60, 98, 85)

        m_params = MansonilInputs(
            at_risk_pop=m_pop,
            pct_heavy=m_heavy_pct / 100,
            pct_light=1 - m_heavy_pct / 100,
            pct_hepatomegaly=m_hepatomeg_pct / 100,
            morbidity_reduction_hepatic=m_morbidity_red / 100,
            cure_rate=m_cure / 100,
        )
        m_caseloads = estimate_caseloads_mansoni(m_pop, m_prev, m_params)

        with st.expander("Estimated baseline caseloads (S. mansoni) — point estimates"):
            cl_df = pd.DataFrame([{
                "Infected": m_caseloads["infected"],
                "Anemia": m_caseloads["anemia"],
                "Hepatomegaly": m_caseloads["hepatomegaly"],
                "Periportal fibrosis": m_caseloads["fibrosis"],
                "Portal hypertension": m_caseloads["portal_htn"],
                "Esophageal varices": m_caseloads["varices"],
            }])
            st.dataframe(cl_df.style.format("{:,.0f}"))
    else:
        m_params = MansonilInputs()
        m_prev   = 0.0
        m_pop    = 0.0

    if run_haematobium:
        st.markdown("---")
        st.subheader("S. haematobium — disease inputs")
        col1, col2 = st.columns(2)
        with col1:
            h_prev = st.number_input(
                "S. haematobium SAC prevalence (%)",
                value=round(prev_sac, 1), min_value=0.0, max_value=100.0,
            )
            h_pop = st.number_input(
                "At-risk population (S. haematobium)",
                value=float(pop_req_mda),
            )
            h_female_pct = st.slider("% female in at-risk population", 30, 70, 50)
        with col2:
            h_bg_cancer = st.number_input(
                "Background bladder cancer rate (per 100,000)", value=3.5
            )
            h_morbidity_red = st.slider(
                "Urinary morbidity reduction with MDA (%)", 30, 90, 75
            )
            h_cure = st.slider("PZQ cure rate – S. haematobium (%)", 65, 98, 87)

        h_params = HaematobiumInputs(
            at_risk_pop=h_pop,
            female_fraction=h_female_pct / 100,
            bg_bladder_cancer_rate=h_bg_cancer,
            morbidity_reduction_urinary=h_morbidity_red / 100,
            cure_rate=h_cure / 100,
        )
        h_caseloads = estimate_caseloads_haematobium(
            h_pop, h_prev, h_params, h_female_pct / 100
        )

        with st.expander("Estimated baseline caseloads (S. haematobium) — point estimates"):
            cl_h_df = pd.DataFrame([{
                "Infected": h_caseloads["infected"],
                "Hematuria": h_caseloads["hematuria"],
                "Hydronephrosis": h_caseloads["hydronephrosis"],
                "FGS": h_caseloads["fgs"],
                "Bladder cancer (total)": h_caseloads["bladder_cancer_total"],
                "PAF (%)": h_caseloads["paf"] * 100,
            }])
            st.dataframe(cl_h_df.style.format({
                **{c: "{:,.1f}" for c in cl_h_df.columns if c != "PAF (%)"},
                "PAF (%)": "{:.2f}%",
            }))
    else:
        h_params = HaematobiumInputs()
        h_prev   = 0.0
        h_pop    = 0.0
        h_female_pct = 50

# ── RUN PSA (cached) ──────────────────────────────────────────────────────────
N_ITER = 1_000
sim_m, sim_h = None, None

if run_mansoni and m_pop > 0 and m_prev > 0:
    sim_m = run_monte_carlo_mansoni(N_ITER, m_pop, m_prev, f"{m_pop}_{m_prev}_{m_heavy_pct}")

if run_haematobium and h_pop > 0 and h_prev > 0:
    sim_h = run_monte_carlo_haematobium(
        N_ITER, h_pop, h_prev, h_female_pct / 100, life_exp, f"{h_pop}_{h_prev}_{h_female_pct}"
    )

# ── TAB 3: Results ────────────────────────────────────────────────────────────
with tabs[3]:
    if sim_m is None and sim_h is None:
        st.info("Set disease prevalence and population in the **Disease inputs** tab to generate results.")
        st.stop()

    # ── Combined DALY summary ────────────────────────────────────────────────
    st.subheader("DALY burden: no-MDA vs MDA scenario")

    combined_icer = None

    for label, sim_df, species_tag in [
        ("S. mansoni / japonicum", sim_m, "mansoni"),
        ("S. haematobium", sim_h, "haematobium"),
    ]:
        if sim_df is None:
            continue
        with st.expander(f"**{label}** — DALY breakdown", expanded=True):
            daly_df = daly_summary_table(sim_df, species_tag)
            fmt_cols = {c: "{:,.1f}" for c in daly_df.columns if daly_df[c].dtype == float}
            st.dataframe(daly_df.style.format(fmt_cols))
            st.altair_chart(plot_daly_breakdown(daly_df), use_container_width=True)

    # ── Productivity losses ──────────────────────────────────────────────────
    st.subheader("Productivity losses")
    for label, sim_df in [("S. mansoni", sim_m), ("S. haematobium", sim_h)]:
        if sim_df is None:
            continue
        prod = productivity_summary(sim_df, daily_wage_adj)
        with st.expander(f"{label} — productivity"):
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Workdays lost p.a. (no MDA)",
                f"{prod['mean_no']:,.0f}",
                help=_format_ci(prod["mean_no"], prod["lo_no"], prod["hi_no"]),
            )
            col2.metric(
                "Workdays lost p.a. (MDA)",
                f"{prod['mean_mda']:,.0f}",
                help=_format_ci(prod["mean_mda"], prod["lo_mda"], prod["hi_mda"]),
            )
            col3.metric("Productivity days gained p.a.", f"{prod['days_gained']:,.0f}")
            st.write(
                f"Estimated annual economic gain from MDA: "
                f"**${prod['econ_gain_pa']:,.0f}** "
                f"(discounted over {time_horizon} yr: "
                f"**${prod['econ_gain_disc']:,.0f}**). "
                f"Wages adjusted for bottom quintile income share ({q1_share*100:.1f}%)."
            )

    # ── Health sector costs ──────────────────────────────────────────────────
    st.subheader("Health sector costs")
    total_hs_savings = 0.0
    total_econ_gain  = 0.0
    total_dalys_av   = 0.0

    for label, sim_df in [("S. mansoni", sim_m), ("S. haematobium", sim_h)]:
        if sim_df is None:
            continue
        hsc = health_sector_costs(sim_df, opd_cost_curr, ipd_cost_curr)
        prod = productivity_summary(sim_df, daily_wage_adj)
        total_hs_savings += hsc["hs_savings_pa"]
        total_econ_gain  += prod["econ_gain_pa"]

        with st.expander(f"{label} — health sector costs"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Annual HS cost (no MDA)", f"${hsc['hs_cost_no']:,.0f}")
            c2.metric("Annual HS cost (MDA)", f"${hsc['hs_cost_mda']:,.0f}")
            c3.metric("Annual HS savings", f"${hsc['hs_savings_pa']:,.0f}")
            st.caption(f"Discounted savings over {time_horizon} yr: ${hsc['hs_savings_disc']:,.0f}")

    # ── ICER and ROI ────────────────────────────────────────────────────────
    st.subheader("Cost-effectiveness")

    # Use combined DALY dataframe if both species run
    for label, sim_df in [
        ("S. mansoni", sim_m),
        ("S. haematobium", sim_h),
    ]:
        if sim_df is None:
            continue
        icer_res = compute_icer(sim_df, annual_prog_cost)
        total_dalys_av += icer_res["dalys_averted_mean"]

        with st.expander(f"{label} — ICER and CEAC", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "DALYs averted p.a.",
                f"{icer_res['dalys_averted_mean']:,.0f}",
                help=_format_ci(
                    icer_res["dalys_averted_mean"],
                    icer_res["dalys_averted_lo"],
                    icer_res["dalys_averted_hi"],
                ),
            )
            c2.metric(
                "ICER (USD/DALY)",
                f"${icer_res['icer_mean']:,.0f}",
                help=_format_ci(icer_res["icer_mean"], icer_res["icer_lo"], icer_res["icer_hi"],
                                 fmt=",.0f"),
            )
            c3.metric(
                "% prob. cost-effective (Woods CET)",
                f"{icer_res['pct_cost_effective_cet']*100:.0f}%",
            )
            st.write(
                f"The Woods et al. CEA threshold for {country} is "
                f"**${cet:,.0f}** per DALY averted. "
                f"The GDP-based threshold (1× GDP) is **${annual_ppp:,.0f}**. "
                f"The ICER is cost-effective under "
                f"{'both' if icer_res['icer_mean'] < min(cet, annual_ppp) else 'at least one'} "
                f"threshold definition."
            )
            st.altair_chart(
                plot_ceac(icer_res["ceac_wtp"], icer_res["ceac_prob"], cet),
                use_container_width=True,
            )

    # ── ROI narrative ────────────────────────────────────────────────────────
    st.subheader("Return on investment")
    roi = compute_roi(total_hs_savings, total_econ_gain, annual_prog_cost)
    st.markdown(
        f"""
        For every **$1** invested in the schistosomiasis MDA programme at this
        administrative level, the estimated economic return is **${roi:.2f}**,
        combining health sector cost savings (${total_hs_savings:,.0f} p.a.) and
        inequality-adjusted productivity gains (${total_econ_gain:,.0f} p.a.).
        This is a conservative estimate: it excludes educational benefits,
        caregiver time savings, and spillover effects on co-administered drugs
        for other NTDs.
        """
    )

    # ── Download raw PSA data ────────────────────────────────────────────────
    with st.expander("Download raw Monte Carlo simulation data"):
        if sim_m is not None:
            download_df(sim_m, "Download S. mansoni PSA data (CSV)")
        if sim_h is not None:
            download_df(sim_h, "Download S. haematobium PSA data (CSV)")

# ── TAB 4: Budget Impact ─────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Budget impact analysis")
    st.caption(
        f"Discounted {bia_horizon}-year projection | "
        f"Discount rate: {disc_costs:.0%} | "
        f"MDA frequency: {mda_frequency}"
    )

    if sim_m is None and sim_h is None:
        st.info("Complete the Disease inputs tab first.")
    else:
        _hs_savings = sum(
            health_sector_costs(sim_df, opd_cost_curr, ipd_cost_curr)["hs_savings_pa"]
            for sim_df in [sim_m, sim_h] if sim_df is not None
        )
        _econ_gain = sum(
            productivity_summary(sim_df, daily_wage_adj)["econ_gain_pa"]
            for sim_df in [sim_m, sim_h] if sim_df is not None
        )
        bia_df = budget_impact_analysis(
            annual_prog_cost=annual_prog_cost,
            hs_savings_pa=_hs_savings,
            econ_gain_pa=_econ_gain,
            horizon=bia_horizon,
            disc_rate=disc_costs,
            pzq_cost=pzq_unit_cost,
            pop_treat=pop_trt_mda,
            pzq_per_person=pzq_per_person,
            delivery_c=delivery_cost,
            freq=mda_frequency,
        )
        st.altair_chart(plot_bia(bia_df), use_container_width=True)

        st.markdown("---")
        fmt = {
            "Drug_costs_USD": "${:,.0f}",
            "Delivery_costs_USD": "${:,.0f}",
            "Other_prog_USD": "${:,.0f}",
            "Total_prog_cost_USD": "${:,.0f}",
            "Health_sector_savings_USD": "${:,.0f}",
            "Economic_gains_USD": "${:,.0f}",
            "Net_benefit_USD": "${:,.0f}",
            "Cumulative_net_benefit_USD": "${:,.0f}",
        }
        styled_table(bia_df, fmt)
        download_df(bia_df, "Download BIA table (CSV)")

        # Summary metrics
        total_cost  = bia_df["Total_prog_cost_USD"].sum()
        total_ben   = (bia_df["Health_sector_savings_USD"] + bia_df["Economic_gains_USD"]).sum()
        bcr         = total_ben / max(total_cost, 1.0)
        st.metric(
            f"Benefit–cost ratio over {bia_horizon} years",
            f"{bcr:.2f}",
            help="Total discounted benefits / total discounted programme costs",
        )

# ── TAB 5: Technical assumptions ─────────────────────────────────────────────
with tabs[5]:
    st.subheader("Technical assumptions and parameter sources")

    with st.expander("S. mansoni — parameter table"):
        m_tech = [
            ["Light infection (% infected)", "60%", "40–75%", "WHO 2002"],
            ["Anemia — light infection (%)", "18%", "8–28%", "King et al. 2005"],
            ["Anemia — heavy infection (%)", "45%", "30–60%", "King et al. 2005"],
            ["Hepatomegaly (% heavy)", "22%", "10–35%", "van der Werf et al. 2003"],
            ["Periportal fibrosis (% hepatomeg.)", "48%", "30–65%", "van der Werf et al. 2003"],
            ["Portal HTN (% fibrosis)", "32%", "18–48%", "van der Werf et al. 2003"],
            ["Esophageal varices (% portal HTN)", "58%", "40–75%", "Richter et al. 2001"],
            ["DW — mild anemia", "0.006", "0.001–0.012", "GBD 2019"],
            ["DW — moderate anemia", "0.058", "0.030–0.095", "GBD 2019"],
            ["DW — hepatomegaly", "0.021", "0.009–0.037", "GBD 2019"],
            ["DW — ascites / portal HTN", "0.222", "0.145–0.308", "GBD 2019"],
            ["DW — esophageal varices", "0.190", "0.117–0.273", "GBD 2019"],
            ["PZQ cure rate", "85%", "70–95%", "Zwang & Olliaro 2014"],
            ["Egg reduction rate (ERR)", "90%", "80–97%", "Zwang & Olliaro 2014"],
            ["Hepatic morbidity reduction (MDA)", "70%", "50–85%", "Garba et al. 2013"],
            ["Productivity loss — anemia", "12%", "4–22%", "Audibert et al. 1999"],
            ["Productivity loss — varices/portal HTN", "65%", "45–80%", "Croce et al. 2010"],
        ]
        st.table(pd.DataFrame(m_tech, columns=["Parameter", "Base value", "Range", "Source"]))

    with st.expander("S. haematobium — parameter table"):
        h_tech = [
            ["Hematuria (% infected)", "62%", "45–78%", "WHO 2002"],
            ["Hydronephrosis (% infected)", "15%", "8–24%", "Poggensee & Feldmeier 2001"],
            ["FGS (% infected females)", "75%", "60–88%", "Hotez & Kamath 2009"],
            ["Bladder cancer RR", "4.2", "2.5–7.0", "IARC Monograph 2012"],
            ["Background bladder cancer (per 100k)", "3.5", "2.0–6.0", "GLOBOCAN 2020"],
            ["DW — hematuria", "0.020", "0.008–0.035", "GBD 2019"],
            ["DW — hydronephrosis", "0.149", "0.093–0.213", "GBD 2019"],
            ["DW — FGS", "0.048", "0.026–0.076", "GBD 2019"],
            ["DW — bladder cancer primary", "0.288", "0.185–0.399", "GBD 2019"],
            ["DW — bladder cancer metastatic", "0.540", "0.400–0.680", "GBD 2019"],
            ["PZQ cure rate", "87%", "72–96%", "Zwang & Olliaro 2014"],
            ["Urinary morbidity reduction (MDA)", "75%", "55–90%", "Garba et al. 2013"],
            ["Cancer risk reduction (MDA)", "30%", "12–52%", "Botelho et al. 2011"],
            ["Productivity loss — hematuria", "8%", "2–16%", "Author estimate"],
            ["Productivity loss — hydronephrosis", "25%", "12–40%", "Author estimate"],
            ["Productivity loss — cancer", "75%", "55–90%", "Croce et al. 2010"],
        ]
        st.table(pd.DataFrame(h_tech, columns=["Parameter", "Base value", "Range", "Source"]))

    with st.expander("Economic methodology"):
        st.markdown("""
        **Productivity loss approach** — Human capital method with inequality adjustment.
        Daily wage = GDP(PPP) × bottom-quintile income share / (0.20 × 300 working days).
        Reference: Mathew et al. (2020); Ramaiah & Guruswamy (2008).

        **CEA threshold** — Woods et al. (2016) elasticity method (elasticity = 2.478,
        UK reference CET = £12,936 in 2013 prices). Preferred over 1-3× GDP (no empirical basis).

        **Discounting** — 3% for both costs and effects (WHO reference case).
        Sensitivity: 0% and 5% available via sidebar.

        **Bladder cancer DALY** — includes both YLD (years lived with disability) and
        YLL (years of life lost). YLL = N_cancer_deaths × (LE – mean diagnosis age 55).
        Fatality fractions: primary 35% (5-year), metastatic 85% (2-year).

        **Budget impact** — Discounted present value of programme costs vs combined
        health-sector savings and productivity gains. BCR = total benefits / total costs.
        """)

    with st.expander("Distribution assumptions (PSA)"):
        st.markdown("""
        | Parameter type | Distribution | Rationale |
        |---|---|---|
        | Proportions (%, 0-1) | Beta (mean, SD) | Bounded 0-1 |
        | Efficacy (cure rate, ERR) | Beta | Bounded 0-1 |
        | Disability weights | Truncated normal | Symmetric uncertainty, always positive |
        | Relative risks | Gamma | Right-skewed, always positive |
        | Costs | Log-normal | Right-skewed cost distributions |
        | Productivity losses | Truncated normal | Symmetric, positive |

        All uncertainty intervals are **90% (5th–95th percentile)** from 1,000 iterations.
        """)

# ── TAB 6: Contact ────────────────────────────────────────────────────────────
with tabs[6]:
    with st.expander("Contact the development team"):
        contact_form = """
        <form action="https://formsubmit.co/your@email.com" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required><br><br>
            <input type="email" name="email" placeholder="Your email" required><br><br>
            <input type="text" name="_honey" style="display:none">
            <textarea name="message" placeholder="Your message or question" rows="5"></textarea><br><br>
            <button type="submit">Send</button>
        </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)

    st.markdown("""
    **Bug reports and feature requests**: please open an issue on the
    [GitHub repository]([URL]).

    **Citation**: Please cite the accompanying manuscript when using outputs in
    publications or reports. BibTeX entry available in the repository README.
    """)
