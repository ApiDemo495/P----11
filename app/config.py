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

# Weighted CCI summands: core signal formulas + brain consensus + CERC.
#   F20 (DRHE)     -> multiplier only (not a directional signal)
#   F28 (ARC)      -> risk management, not a signal ("not a signal formula")
#   F29 (DLP)      -> level placement (price outputs, not [-1,1])
#   F30 (PCV)      -> validator: needs CCI as input (post-CCI layer)
#   F21 (CCI)      -> the aggregator itself
CORE_SIGNAL_IDS: list[int] = list(range(1, 20)) + [22, 23, 24, 25, 26, 27, 31]
BRAIN_COMPONENT_ID = 97        # brain_consensus weighted input to CCI
EMOTION_COMPONENT_ID = 98      # CERC weighted input to CCI
WEIGHT_IDS: list[int] = CORE_SIGNAL_IDS + [BRAIN_COMPONENT_ID, EMOTION_COMPONENT_ID]
N_WEIGHT_COMPONENTS = len(WEIGHT_IDS)   # 28

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
