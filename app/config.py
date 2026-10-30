"""Global configuration. Every constant mirrors the specification exactly."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Master clock — World Clock Alignment System
# ---------------------------------------------------------------------------
GRC_INTERVAL_SECONDS = 4 * 60          # 4-minute intervals -> 15 GRCs per hour
GRC_PER_HOUR = 15
INITIAL_ANALYSIS_MAX_SECONDS = 4 * 60  # Initial Analysis Phase cap: 4 minutes

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
SYMBOLS: dict[str, str] = {
    "BTC": "BTCUSDT",
    "PAXG": "PAXGUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
}
SYMBOL_ORDER: list[str] = ["BTC", "PAXG", "ETH", "BNB"]

# ---------------------------------------------------------------------------
# Prediction timeframes and sub-targets
# ---------------------------------------------------------------------------
TIMEFRAMES: dict[str, dict] = {
    "tf1": {"minutes": 22, "targets": [5, 10, 15, 22]},
    "tf2": {"minutes": 44, "targets": [11, 22, 33, 44]},
    "tf3": {"minutes": 130, "targets": [30, 40, 90, 130]},
    "tf4": {"minutes": 247, "targets": [60, 120, 180, 247]},
}
TF_ORDER: list[str] = ["tf1", "tf2", "tf3", "tf4"]

# Lock strengths (Composite Confidence Index shift required to adjust a locked sub-target)
LOCK_SHIFT_ADJUST: dict[str, float | None] = {
    "tp1": 0.0,    # lowest lock strength — adjusts at every GRC before being hit
    "tp2": 0.3,    # medium lock strength
    "tp3": 0.5,    # high lock strength
    "tp4": None,   # macro thesis — changes only under Extreme Override Conditions
}
EXTREME_OVERRIDE_CCI = 0.25       # Confidence_final must drop below this
EXTREME_OVERRIDE_NEWS = 0.8       # opposing News Impact Aggregate must exceed this

# ---------------------------------------------------------------------------
# Composite Confidence Index (Formula 21)
# ---------------------------------------------------------------------------
N_FORMULAS = 21                   # Formulas 1..21 exist in this part; 22..31 arrive with part 2
# Formulas entering the CCI weighted sum:
#   F1..F19 (part 1) + F22..F31 (part 2)  = 29 eventual terms.
#   F20 (DRHE) is EXCLUDED from the sum: it is explicitly "not a directional
#   signal", is not bounded to [-1,+1], and enters only through
#   Confidence_final = |CCI_adjusted| * min(1, DRHE).
#   F21 is the aggregator itself.
N_FORMULAS_ACTIVE = 19
LEARNING_RATE_ETA = 0.02
WEIGHT_MIN = 0.005
WEIGHT_MAX = 0.15
CONFIDENCE_FLOOR = 0.35           # below -> NO TRADE
DOUBLE_CHECK_RATIO = 0.6
DOUBLE_CHECK_PENALTY = 0.5

# ---------------------------------------------------------------------------
# Section 10 — auto-learning timescales
# ---------------------------------------------------------------------------
ETA_MICRO = 0.02                  # core formula weights (every GRC)
ETA_BRAIN = 0.01                  # per-brain influence weights (every GRC)
ETA_EMOTION = 0.015               # CERC member weights (every GRC)
ETA_PHYSICS = 0.015               # physics alpha weights (every GRC)
BRAIN_W_MIN, BRAIN_W_MAX = 0.1, 0.4           # per-brain influence clamp
EMOTION_W_MIN, EMOTION_W_MAX = 0.01, 0.12     # CERC member clamp
PHYSICS_W_MIN, PHYSICS_W_MAX = 0.01, 0.12     # physics alpha clamp
MACRO_PERIOD_GRC = 15             # every 60 minutes
MACRO_ACC_WINDOW = 15             # rolling accuracy window (GRCs)
MACRO_PROBATION = 0.4             # below -> halve weight + flag
MACRO_BONUS = 0.7                 # above -> +20% weight
MACRO_CORR_REDUNDANCY = 0.85      # pairwise correlation above -> cut 30%
MACRO_CUT = 0.30
DEEP_PERIOD_MINUTES = 247         # one full TF4 cycle
DEEP_DIAGNOSTIC_HR = 0.5          # HR below -> diagnostic mode
DEEP_WRONG_CUT = 0.40             # top-3 wrong formulas: -40%
DEEP_RIGHT_BOOST = 0.50           # rescued formulas: +50%
ACCURACY_TARGET = 0.90
HIGH_ALERT_LOW = 0.85             # rolling-100 accuracy below -> double rates
HIGH_ALERT_HIGH = 0.92            # above -> return to normal rates
HIGH_ALERT_PROBATION = 0.5        # probation threshold while in high-alert
ACCURACY_ROLLING = 100            # per-timeframe rolling window

# Section 14 Level 1 — subsystem weightages of the CCI
SUBSYSTEM_SHARES_INIT: dict[str, float] = {
    "core": 0.45, "brain": 0.10, "emotion": 0.15,
    "physics": 0.15, "news": 0.07, "hf": 0.08,
}
SUBSYSTEM_BEARINGS: dict[str, tuple[float, float]] = {
    "core": (0.05, 0.60), "brain": (0.05, 0.25), "emotion": (0.05, 0.25),
    "physics": (0.05, 0.25), "news": (0.03, 0.15), "hf": (0.03, 0.12),
}
SUBSYSTEM_STEP = 0.02             # Level-1 adjustment step (Section 14)

# Section 11 — agent integration
AGENT_GEMINI_WEIGHT = 0.10
AGENT_LOCAL_WEIGHT = 0.08
AGENT_COMBINED_WEIGHT = 0.12
AGENT_FREE_RPH = 15               # free-tier requests per hour (1 / GRC)
AGENT_PAID_PER_GRC = 4            # per-asset calls when paid tier detected
AGENT_FALLBACK_GRCS = 3           # reuse last signal for 3 failed GRCs
AGENT_EPS = 1e-6

# Section 13 — history retention
HISTORY_DIR = "data/history"
HISTORY_RETENTION_DAYS = 90
LEARN_STATE_FILE = "data/auto_learning_state.json"

# ---------------------------------------------------------------------------
# Bgn synchronization
# ---------------------------------------------------------------------------
BGN_BETA = 1.2
BGN_RPIC_HISTORY = 50             # coherence over last 50 GRC cycles
BGN_MIN_COHERENCE_POINTS = 3      # below this, coherence defaults to 1 (bootstrap)

# ---------------------------------------------------------------------------
# Formula history windows (number of GRC cycles)
# ---------------------------------------------------------------------------
HIST_GAMMA_MAX = 500              # Gamma_max over last 500 GRC cycles (F1)
HIST_IRECENT_MAX = 200            # max inversions over last 200 GRCs (F2)
HIST_SS = 100                     # self-similarity baseline (F6)
HIST_CWP = 200                    # sigma_CWP (F7)
HIST_TAU = 300                    # tau_half baseline (F9)
HIST_IDIV = 300                   # I_div_max (F10)
HIST_SA = 300                     # SA_max (F16)
HIST_STD_R = 200                  # volatility baseline (F20)
HIST_REGIME = 500                 # regime classification history (F20)
HIST_REGIME_DURATIONS = 500       # completed durations per regime type (F20)
HIST_CCMDR_STREAKS = 200          # completed (length, lambda) streaks (F18)
HIST_CCMDR_WINDOW = 50            # last 50 streaks similar in length feed lambda_critical (F18)

# ---------------------------------------------------------------------------
# Risk system (spec Section 1 — exact mapping; implementation seam for F28/F29 part 2)
# ---------------------------------------------------------------------------
RATIO_3_TO_1 = 3.0
RATIO_3_TO_2 = 1.5
RATIO_1_TO_1 = 1.0
SL_TIGHTEN_FACTOR = 0.15          # SL tightened by 15% after SAFE event
SL_CONFIDENCE_PENALTY = 0.15      # confidence reduced by 0.15 after SAFE event

# ---------------------------------------------------------------------------
# Provisional projection / level constants (replaced by exact F28/F29 in part 2)
# ---------------------------------------------------------------------------
TARGET_SCALE = 0.5                # provisional sub-target projection scale
LEVEL_BASE_SCALE = 0.25           # provisional TP/SL base distance scale (F29 seam)

# ---------------------------------------------------------------------------
# Data feed
# ---------------------------------------------------------------------------
BINANCE_REST = "https://api.binance.com"
BINANCE_WS = "wss://stream.binance.com:9443/stream"
BACKFILL_BARS = 1000              # 1m klines backfilled at startup
MIN_BARS_FULL_WINDOW = TIMEFRAMES["tf4"]["minutes"] + 1   # 248
MIN_BARS_COMPUTE = 30             # below this a formula pass returns zeros
FEED_MODE = "auto"                # auto | binance | sim   (auto falls back to simulator)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = "data"
STATE_FILE = "data/state.json"

# ---------------------------------------------------------------------------
# PART 2 — Formulas 22-31, Drosophila Brain system, 22 Emotion formulas
# ---------------------------------------------------------------------------

# Weighted CCI summands: core signal formulas + brain consensus + CERC
# (part 2), then physics UFP + news aggregate + hedge/dark-pool combined
# (part 3) -> 31 components.
#   F20 (DRHE)     -> multiplier only (not a directional signal)
#   F28 (ARC)      -> risk management, not a signal ("not a signal formula")
#   F29 (DLP)      -> level placement (price outputs, not [-1,1])
#   F30 (PCV)      -> validator: needs CCI as input (post-CCI layer)
#   F21 (CCI)      -> the aggregator itself
CORE_SIGNAL_IDS: list[int] = list(range(1, 20)) + [22, 23, 24, 25, 26, 27, 31]
BRAIN_COMPONENT_ID = 97        # brain_consensus weighted input to CCI
EMOTION_COMPONENT_ID = 98      # CERC weighted input to CCI
PHYSICS_COMPONENT_ID = 99      # P22 Unified Field Prediction (Bgn-synced)
NEWS_COMPONENT_ID = 100        # News Impact Aggregate (Bgn-synced)
HEDGE_COMPONENT_ID = 101       # HF1+HF2+HF3 combined (Bgn-synced)
WEIGHT_IDS: list[int] = (
    CORE_SIGNAL_IDS
    + [BRAIN_COMPONENT_ID, EMOTION_COMPONENT_ID,
       PHYSICS_COMPONENT_ID, NEWS_COMPONENT_ID, HEDGE_COMPONENT_ID]
)
N_WEIGHT_COMPONENTS = len(WEIGHT_IDS)   # 31
# Section 8: "Their initial combined weight is 0.08 (8% of total CCI)."
HEDGE_INITIAL_WEIGHT = 0.08
# P22: physics layer's total contribution weighted by TME (P9):
# effective_omega(UFP) = omega(UFP) * TME(t)  (stored weight is the base)

# Formula 30 — PCV validator
PCV_CONFIDENCE_REDUCTION = 0.3   # when PCV < 0: confidence -= |PCV| * 0.3

# Emotion E9 — RTP confidence reduction
RTP_CONFIDENCE_REDUCTION = 0.2   # when RTP > 0.6: confidence -= RTP * 0.2

# Brain 2 — Experience Quality soft modifier: CCI * (0.7 + 0.3 * EQ)
EXPERIENCE_BASE, EXPERIENCE_RANGE = 0.7, 0.3

# Brain 3 — dominant market-state confidence boost
STATE_CONFIDENCE_BOOST = 0.1

# Brain 4 — secondary (slower) weight adjustment on top of CCI reward rule
BRAIN4_WEIGHT_LEARNING = 0.01

# Reduced-connectome optimization (spec: 5,000 individual + ~2,700 clusters)
BRAIN_INDIVIDUAL_UNITS = 5000
BRAIN_CLUSTER_UNITS = 2700
BRAIN_CLUSTER_SIZE = 50
BRAIN_OUT_DEGREE = 4             # sparse synthetic edge fan-out (perf budget)
BRAIN_TOTAL_UNITS = BRAIN_INDIVIDUAL_UNITS + BRAIN_CLUSTER_UNITS   # 7,700

# Named Drosophila populations (spec Section 4)
POP_SIZES = {
    "PAM": 130,      # dopaminergic (dopamine input)
    "SE": 80,        # serotonergic
    "OA": 40,        # octopaminergic
    "TA": 30,        # tyraminergic
    "MBON": 34,      # decision output layer
    "KC": 2000,      # kenyon cells (pattern recognition)
    "LH": 1500,      # lateral horn (innate valence)
    "CC": 250,       # central complex (state/navigation)
    "FSB": 150,      # fan-shaped body (working memory)
    "EB": 100,       # ellipsoid body (adaptive behavior)
    "DN": 500,       # descending neurons (motor/weight commands)
}
BRAIN_MARKET_STATES = ["accumulation", "distribution", "markup", "markdown", "ranging"]

# neuPrint (optional): set NEUPRINT_URL / NEUPRINT_TOKEN to attempt a live
# connectome fetch; otherwise a deterministic structural fallback is used.
NEUPRINT_URL = ""                # e.g. https://connectome-neo4j.janelia.org
NEUPRINT_TOKEN = ""

# Formula 27 — volatility-cluster auto-learning history
HIST_VOL_CLUSTERS = 200

# Formula 31 — retrodiction anomaly baselines
HIST_R_ANOMALY = 200

# E8 — GRC-lagged emotion history
HIST_EMOTION_LAG = 10

# E21 — composite emotion volatility over last 10 GRCs
HIST_COMPOSITE_EMOTION = 10

# ---------------------------------------------------------------------------
# PART 3 — Section 6: 22 Physics Formulas (Market Quantum State Framework)
# ---------------------------------------------------------------------------
PHYSICS_ALPHA_COUNT = 21            # alpha_p weights inside P22 UFP (1/21 each)
UFP_ALPHA_IDS: list[int] = list(range(1, 22))
PHYSICS_NAMES: dict[int, str] = {
    1: "PSSA", 2: "WFCP", 3: "HPVU", 4: "QTPB", 5: "MEE", 6: "DRPT",
    7: "BEOFC", 8: "FDPD", 9: "TME", 10: "PIPP", 11: "RGVF",
    12: "SSBD", 13: "GLPLO", 14: "CELG", 15: "DSMS", 16: "RTDMP",
    17: "BBIR", 18: "TPTI", 19: "GSPI", 20: "PDOBV", 21: "QDPC",
    22: "UFP",
}
PHYSICS_DISPLAY_IDS: list[int] = list(range(1, 23))   # P1..P22 for panels

QUANTUM_LEVELS = 21                # P1: N = 21 (odd, centered on price)
TAU_REF_MINUTES = 60.0             # P1 sigma scaling + P11 RGVF tau_ref
PIPP_PATHS = 1000                  # P10 Monte Carlo K
PIPP_MAX_STEPS = 20                # P10 M = min(20, tau)
QTPB_INTEGRAL_POINTS = 20          # P4 numerical integral points
PROFILE_BINS = 100                 # P13 volume profile bins
LVD_VACUUM_RATIO = 0.5             # P1 LVD: local volume ratio ramp anchor
TME_HOT_VOLUME_MULT = 2.0          # P9 hot minute: V > 2 * V_avg
TME_COLD_VOLUME_MULT = 0.5         # P9 cold minute: V < 0.5 * V_avg
TME_NEWS_GATE = 0.5                # P9 hot minute: news_impact(i) > 0.5
TME_NEWS_PRICE_GATE = 0.3          # P21 significant news event: > 0.3
SSBD_NEIGHBORS = 2                 # P12 z = 2 (Ising neighbours)
QDPC_K_B = 1.0                     # P21 Boltzmann constant set to 1
BBIR_K_B = 1.0                     # P17 Boltzmann constant set to 1
PHYSICS_COUPLING_WINDOW = 50       # P22 coupling: corr over last 50 GRCs
HIST_TPTI = 200                    # P18 beta1 baseline over last 200 GRCs
HIST_T_BASELINE = 200              # P17 T_baseline over last 200 GRCs
C_MARKET_WINDOW = 500              # P16 speed of light: max |r1min| / 500 min
C_MARKET_LEAD_WINDOW = 200         # P15 speed of light: / 200 min across assets

# ---------------------------------------------------------------------------
# PART 3 — Section 7: News categories, base weightages, impact durations
# ---------------------------------------------------------------------------
NEWS_CATEGORIES: dict[str, dict] = {
    "regulatory":   {"weight": 0.25, "duration": 247},   # TF4 full window
    "institutional": {"weight": 0.20, "duration": 130},  # TF3 window
    "protocol":     {"weight": 0.15, "duration": 44},    # TF2 window
    "macro":        {"weight": 0.15, "duration": 130},   # TF3 window
    "structure":    {"weight": 0.10, "duration": 44},    # TF2 window
    "sentiment":    {"weight": 0.08, "duration": 22},    # TF1 window
    "geopolitical": {"weight": 0.07, "duration": 247},   # TF4, low immediate
}
NEWS_FRESHNESS_RATE = 0.05         # Freshness = exp(-0.05 * age_minutes)
NEWS_AGGREGATE_WINDOW = 247        # sum over the last 247 minutes, clamp [-1,1]
# Source_Reliability tiers (Section 7)
NEWS_RELIABILITY_TOP = 1.0         # Reuters, Bloomberg, CoinDesk
NEWS_RELIABILITY_MID = 0.7         # crypto blogs, verified accounts
NEWS_RELIABILITY_LOW = 0.4         # unverified social media, forums
NEWS_TOP_SOURCES = {
    "reuters", "bloomberg", "coindesk", "wsj", "ft", "cnbc",
}
NEWS_MID_SOURCES = {
    "cointelegraph", "theblock", "the block", "coingeek", "decrypt",
    "messari", "glassnode", "cryptoquant", "twitter", "x",
}
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_API_KEY = ""                  # optional live key -> falls back to simulator
NEWS_POLL_SECONDS = 60             # live poll cadence
NEWS_QUERY = "bitcoin OR ethereum OR crypto"
PHYSICS_STATE_FILE = "data/physics_weights.json"
