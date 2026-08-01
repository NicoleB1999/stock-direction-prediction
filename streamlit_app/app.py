from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "master_dataset.csv"
MODEL_PATH = BASE_DIR / "best_random_forest_model.pkl"

FEATURES = [
    "open_price",
    "adj_close",
    "volume",
    "FED_RATE",
    "INFLATION_CPI",
    "CONSUMER_CONFIDENCE",
    "SP500",
    "VIX",
    "US10Y",
]

STOCK_LABELS = {
    "AAPL": "Apple",
    "GOOGL": "Alphabet (Google)",
    "NVDA": "NVIDIA",
}

FEATURE_LABELS = {
    "open_price": "מחיר פתיחה",
    "adj_close": "מחיר סגירה מתואם",
    "volume": "נפח מסחר",
    "FED_RATE": "ריבית הפד",
    "INFLATION_CPI": "מדד המחירים לצרכן",
    "CONSUMER_CONFIDENCE": "אמון הצרכנים",
    "SP500": "מדד S&P 500",
    "VIX": "מדד VIX",
    "US10Y": "תשואת אג״ח ארה״ב ל-10 שנים",
}

st.set_page_config(
    page_title="Smart Stock Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --primary: #2563eb;
            --secondary: #7c3aed;
            --accent: #06b6d4;
            --surface: rgba(255, 255, 255, 0.92);
            --border: rgba(37, 99, 235, 0.13);
            --text: #0f172a;
            --muted: #64748b;
        }

        /* רקע כללי ומבנה RTL תקין */
        html, body, [data-testid="stAppViewContainer"] {
            direction: ltr;
            background:
                radial-gradient(circle at 8% 4%, rgba(37, 99, 235, 0.10), transparent 27%),
                radial-gradient(circle at 92% 8%, rgba(124, 58, 237, 0.09), transparent 25%),
                #f7f9fc;
        }

        .block-container,
        [data-testid="stMainBlockContainer"] {
            direction: rtl;
            max-width: 1450px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stMetric"] {
            direction: rtl;
            text-align: right;
        }

        h1, h2, h3, p, label {
            word-break: normal;
            overflow-wrap: break-word;
        }

        /* כותרת ראשית */
        .hero-card {
            position: relative;
            overflow: hidden;
            border-radius: 24px;
            padding: 2rem 2.2rem;
            margin-bottom: 1rem;
            color: white;
            background: linear-gradient(120deg, #1d4ed8 0%, #2563eb 45%, #7c3aed 100%);
            box-shadow: 0 18px 45px rgba(37, 99, 235, 0.22);
        }

        .hero-card::after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            left: -70px;
            top: -110px;
            background: rgba(255,255,255,0.10);
        }

        .hero-kicker {
            font-size: 0.92rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            opacity: 0.9;
            margin-bottom: 0.4rem;
        }

        .hero-card h1 {
            color: white !important;
            font-size: clamp(2rem, 4vw, 3.3rem) !important;
            line-height: 1.05 !important;
            margin: 0 0 0.65rem 0 !important;
            text-align: right !important;
        }

        .hero-card p {
            color: rgba(255,255,255,0.92) !important;
            font-size: 1.05rem;
            max-width: 950px;
            margin: 0;
        }

        .hero-chips {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 1.1rem;
        }

        .hero-chip {
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.22);
            font-size: 0.86rem;
            font-weight: 600;
        }

        .notice-card {
            border: 1px solid rgba(245, 158, 11, 0.24);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin: 0.6rem 0 1.15rem 0;
            background: rgba(255, 251, 235, 0.92);
            color: #92400e;
            box-shadow: 0 5px 18px rgba(148, 64, 14, 0.05);
        }

        /* לשוניות */
        [data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: rgba(255,255,255,0.72);
            padding: 0.45rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        [data-baseweb="tab"] {
            height: 46px;
            border-radius: 12px;
            padding: 0 1rem;
            font-weight: 700;
            color: #475569;
        }

        [aria-selected="true"][data-baseweb="tab"] {
            color: #1d4ed8 !important;
            background: linear-gradient(135deg, rgba(37,99,235,0.12), rgba(124,58,237,0.10));
        }

        [data-baseweb="tab-highlight"] {
            display: none;
        }

        /* כרטיסים */
        .advisor-card {
            border: 1px solid rgba(37, 99, 235, 0.18);
            border-radius: 20px;
            padding: 1.35rem 1.5rem;
            margin: 0.75rem 0 1.1rem 0;
            background: linear-gradient(135deg, rgba(239,246,255,0.96), rgba(245,243,255,0.96));
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.08);
        }

        .advisor-card h3 {
            color: #1d4ed8;
            margin-top: 0;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 700;
        }

        [data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 800;
        }

        /* שדות וכפתורים */
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-testid="stDateInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stFileUploaderDropzone"] {
            border-radius: 12px !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            border: 0;
            border-radius: 12px;
            min-height: 42px;
            font-weight: 800;
            color: white;
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.26);
            color: white;
        }

        /* טבלאות וגרפים */
        [data-testid="stDataFrame"],
        [data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.94);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.35rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
            overflow: hidden;
        }

        hr {
            border-color: rgba(148, 163, 184, 0.24) !important;
        }

        .small-note {
            font-size: 0.9rem;
            opacity: 0.82;
        }

        /* התאמה לחצי מסך ומסכים צרים */
        @media (max-width: 1100px) {
            section[data-testid="stSidebar"] {
                display: none !important;
            }

            [data-testid="stSidebarCollapsedControl"] {
                display: none !important;
            }

            [data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
                gap: 0.8rem !important;
            }

            [data-testid="column"] {
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }

            .block-container,
            [data-testid="stMainBlockContainer"] {
                width: 100% !important;
                max-width: 100% !important;
                padding-left: 0.85rem !important;
                padding-right: 0.85rem !important;
            }

            .hero-card {
                padding: 1.45rem 1.25rem;
                border-radius: 18px;
            }

            [data-baseweb="tab-list"] {
                overflow-x: auto !important;
                flex-wrap: nowrap !important;
                white-space: nowrap !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data.columns = [str(col).strip() for col in data.columns]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    for column in FEATURES + ["stock_return"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.dropna(subset=["date", "symbol"] + FEATURES + ["stock_return"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    data["actual_direction"] = (data["stock_return"] > 0).astype(int)

    # אותה חלוקה כרונולוגית שנעשתה בפרויקט: 70% / 15% / 15%.
    train_end = int(np.floor(len(data) * 0.70))
    validation_end = train_end + (len(data) - train_end) // 2
    data["dataset_split"] = "Train"
    data.loc[train_end:validation_end - 1, "dataset_split"] = "Validation"
    data.loc[validation_end:, "dataset_split"] = "Test"
    return data


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(path)


@st.cache_data(show_spinner=False)
def add_model_predictions(data: pd.DataFrame, model_signature: str) -> pd.DataFrame:
    # model_signature גורם ל-Streamlit לרענן Cache כשהקובץ משתנה.
    del model_signature
    model = load_model(str(MODEL_PATH))
    result = data.copy()
    result["predicted_direction"] = model.predict(result[FEATURES]).astype(int)
    result["predicted_probability"] = model.predict_proba(result[FEATURES])[:, 1]
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_prices(symbols: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch adjusted daily prices from Yahoo Finance. Returns an empty frame on failure."""
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    end_exclusive = end_date + timedelta(days=1)

    for symbol in symbols:
        try:
            history = yf.Ticker(symbol).history(
                start=start_date.isoformat(),
                end=end_exclusive.isoformat(),
                auto_adjust=True,
                actions=False,
            )
            if history.empty or "Close" not in history.columns:
                continue
            one = history.reset_index()[["Date", "Close"]].copy()
            one.columns = ["date", "adj_close"]
            one["date"] = pd.to_datetime(one["date"], errors="coerce").dt.tz_localize(None)
            one["symbol"] = symbol
            one["stock_return"] = one["adj_close"].pct_change()
            frames.append(one.dropna(subset=["date", "adj_close"]))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])


def safe_minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    series = series.fillna(series.median() if series.notna().any() else 0.0)
    if np.isclose(series.max(), series.min()):
        normalized = pd.Series(0.5, index=series.index)
    else:
        normalized = (series - series.min()) / (series.max() - series.min())
    return normalized if higher_is_better else 1 - normalized


def calculate_stock_metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, group in data.groupby("symbol"):
        group = group.sort_values("date").dropna(subset=["adj_close", "stock_return"])
        if len(group) < 2:
            continue

        first_price = float(group["adj_close"].iloc[0])
        last_price = float(group["adj_close"].iloc[-1])
        total_return = last_price / first_price - 1 if first_price else np.nan
        trading_days = max(len(group), 1)
        annualized_return = (
            (1 + total_return) ** (252 / trading_days) - 1
            if pd.notna(total_return) and total_return > -1
            else total_return
        )
        annualized_volatility = float(group["stock_return"].std(ddof=0) * np.sqrt(252))
        cumulative = (1 + group["stock_return"].fillna(0)).cumprod()
        drawdown = cumulative / cumulative.cummax() - 1
        max_drawdown = float(drawdown.min())
        positive_day_rate = float((group["stock_return"] > 0).mean())
        momentum_window = min(20, len(group) - 1)
        recent_momentum = (
            last_price / float(group["adj_close"].iloc[-1 - momentum_window]) - 1
            if momentum_window > 0
            else 0.0
        )
        model_probability = (
            float(group["predicted_probability"].tail(20).mean())
            if "predicted_probability" in group.columns
            else np.nan
        )
        risk_adjusted = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0

        rows.append(
            {
                "symbol": symbol,
                "company": STOCK_LABELS.get(symbol, symbol),
                "total_return": total_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "max_drawdown": max_drawdown,
                "positive_day_rate": positive_day_rate,
                "recent_momentum": recent_momentum,
                "model_probability": model_probability,
                "risk_adjusted": risk_adjusted,
                "observations": len(group),
            }
        )
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()


def score_recommendations(
    metrics: pd.DataFrame,
    risk_level: str,
    horizon: str,
    goal: str,
    volatility_tolerance: str,
    company_style: str,
) -> pd.DataFrame:
    scored = metrics.copy()
    scored["return_score"] = safe_minmax(scored["annualized_return"], True)
    scored["low_vol_score"] = safe_minmax(scored["annualized_volatility"], False)
    scored["high_vol_score"] = safe_minmax(scored["annualized_volatility"], True)
    scored["drawdown_score"] = safe_minmax(scored["max_drawdown"].abs(), False)
    scored["positive_score"] = safe_minmax(scored["positive_day_rate"], True)
    scored["momentum_score"] = safe_minmax(scored["recent_momentum"], True)
    scored["model_score"] = safe_minmax(scored["model_probability"], True)
    scored["risk_adjusted_score"] = safe_minmax(scored["risk_adjusted"], True)

    weights = {
        "return_score": 0.20,
        "low_vol_score": 0.15,
        "high_vol_score": 0.00,
        "drawdown_score": 0.15,
        "positive_score": 0.10,
        "momentum_score": 0.10,
        "model_score": 0.10,
        "risk_adjusted_score": 0.20,
    }

    if goal == "שמירה על ההון":
        weights.update(return_score=0.10, low_vol_score=0.25, drawdown_score=0.25, risk_adjusted_score=0.15)
    elif goal == "צמיחה":
        weights.update(return_score=0.25, low_vol_score=0.15, drawdown_score=0.10, risk_adjusted_score=0.20)
    else:  # מקסום תשואה
        weights.update(return_score=0.35, momentum_score=0.20, high_vol_score=0.05, low_vol_score=0.05)

    if risk_level == "נמוכה":
        weights["low_vol_score"] += 0.10
        weights["drawdown_score"] += 0.10
    elif risk_level == "גבוהה":
        weights["return_score"] += 0.10
        weights["momentum_score"] += 0.05
        weights["high_vol_score"] += 0.05
    else:
        weights["risk_adjusted_score"] += 0.10

    if horizon == "קצר (עד שנה)":
        weights["momentum_score"] += 0.10
        weights["model_score"] += 0.05
    elif horizon == "ארוך (מעל 3 שנים)":
        weights["return_score"] += 0.10
        weights["drawdown_score"] += 0.05
        weights["low_vol_score"] += 0.05
    else:
        weights["risk_adjusted_score"] += 0.05

    if volatility_tolerance == "נמוכה":
        weights["low_vol_score"] += 0.10
        weights["drawdown_score"] += 0.05
    elif volatility_tolerance == "גבוהה":
        weights["return_score"] += 0.05
        weights["high_vol_score"] += 0.05

    total_weight = sum(weights.values())
    weights = {key: value / total_weight for key, value in weights.items()}

    scored["fit_score"] = sum(scored[key] * value for key, value in weights.items())

    # תוספת קטנה ושקופה להעדפת סגנון חברה. היא אינה מחליפה את המדדים הכמותיים.
    preference_bonus = pd.Series(0.0, index=scored.index)
    if company_style == "חברה מבוססת ויציבה יחסית":
        preference_bonus.loc[preference_bonus.index.intersection(["AAPL", "GOOGL"])] = 0.03
    elif company_style == "צמיחה אגרסיבית וחדשנות":
        preference_bonus.loc[preference_bonus.index.intersection(["NVDA"])] = 0.03
    scored["fit_score"] = (scored["fit_score"] + preference_bonus).clip(0, 1)

    return scored.sort_values("fit_score", ascending=False)


def build_recommendation_explanation(symbol: str, row: pd.Series, profile: dict[str, str]) -> str:
    reasons: list[str] = []
    if row["annualized_return"] == row.get("annualized_return"):
        reasons.append(f"תשואה שנתית מחושבת של {row['annualized_return']:.1%} בתקופה שנבחרה")
    reasons.append(f"תנודתיות שנתית של {row['annualized_volatility']:.1%}")
    reasons.append(f"ירידה מרבית היסטורית של {row['max_drawdown']:.1%}")
    reasons.append(f"שיעור ימים חיוביים של {row['positive_day_rate']:.1%}")
    return (
        f"{STOCK_LABELS.get(symbol, symbol)} קיבלה את ציון ההתאמה הגבוה ביותר לפרופיל "
        f"({profile['risk']}, {profile['horizon']}, יעד: {profile['goal']}). "
        + "; ".join(reasons)
        + ". ציון המודל קיבל משקל מוגבל בגלל ביצועי החיזוי המתונים."
    )


def format_metrics_for_display(metrics: pd.DataFrame, include_score: bool = False) -> pd.DataFrame:
    display = metrics.reset_index().copy()
    columns = {
        "symbol": "מניה",
        "company": "חברה",
        "total_return": "תשואה מצטברת",
        "annualized_return": "תשואה שנתית מחושבת",
        "annualized_volatility": "תנודתיות שנתית",
        "max_drawdown": "ירידה מרבית",
        "positive_day_rate": "ימים חיוביים",
        "recent_momentum": "מומנטום 20 ימים",
        "model_probability": "ממוצע הסתברות עלייה (מודל)",
        "fit_score": "ציון התאמה",
        "observations": "מספר תצפיות",
    }
    selected = [
        "symbol",
        "company",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "positive_day_rate",
        "recent_momentum",
    ]
    if "model_probability" in display.columns:
        selected.append("model_probability")
    if include_score and "fit_score" in display.columns:
        selected.append("fit_score")
    selected.append("observations")
    display = display[selected].rename(columns=columns)
    return display


def metric_formats(include_score: bool = False) -> dict[str, str]:
    formats = {
        "תשואה מצטברת": "{:.1%}",
        "תשואה שנתית מחושבת": "{:.1%}",
        "תנודתיות שנתית": "{:.1%}",
        "ירידה מרבית": "{:.1%}",
        "ימים חיוביים": "{:.1%}",
        "מומנטום 20 ימים": "{:.1%}",
        "ממוצע הסתברות עלייה (מודל)": "{:.1%}",
    }
    if include_score:
        formats["ציון התאמה"] = "{:.1%}"
    return formats


def compute_classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["actual_direction"]
    y_pred = frame["predicted_direction"]
    values = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": np.nan,
    }
    if y_true.nunique() == 2:
        values["auc"] = roc_auc_score(y_true, frame["predicted_probability"])
    return values


def style_plotly(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.98)",
        font=dict(family="Arial", color="#334155"),
        title=dict(font=dict(size=20, color="#0f172a"), x=0.98, xanchor="right"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=35, r=35, t=75, b=40),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)", zeroline=False)
    return fig


try:
    project_data = load_data(str(DATA_PATH))
    model_file_signature = f"{MODEL_PATH.stat().st_size}-{MODEL_PATH.stat().st_mtime_ns}"
    model = load_model(str(MODEL_PATH))
    project_data = add_model_predictions(project_data, model_file_signature)
except FileNotFoundError as exc:
    st.error(f"קובץ חסר: {exc.filename}. ודאו שה-CSV וה-PKL נמצאים ליד app.py.")
    st.stop()
except Exception as exc:
    st.error(f"האפליקציה לא הצליחה לטעון את הנתונים או המודל: {exc}")
    st.stop()

min_project_date = project_data["date"].min().date()
max_project_date = project_data["date"].max().date()
test_start_date = project_data.loc[project_data["dataset_split"] == "Test", "date"].min().date()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-kicker">מערכת תומכת החלטה ליועצי השקעות</div>
        <h1>📈 Smart Stock Advisor</h1>
        <p>התאמת חלופה לפרופיל הלקוח, השוואת מניות, הצגת תחזיות וקליטת נתונים חדשים — במקום אחד.</p>
        <div class="hero-chips">
            <span class="hero-chip">AAPL</span>
            <span class="hero-chip">GOOGL</span>
            <span class="hero-chip">NVDA</span>
            <span class="hero-chip">Random Forest</span>
        </div>
    </div>
    <div class="notice-card">
        <strong>שימו לב:</strong> המערכת היא פרויקט אקדמי וכלי תומך החלטה. התוצאות אינן ייעוץ השקעות ואינן הוראת קנייה או מכירה.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("מידע על המערכת")
    st.write(f"טווח נתוני הפרויקט: {min_project_date:%d/%m/%Y}–{max_project_date:%d/%m/%Y}")
    st.write("מניות: AAPL, GOOGL, NVDA")
    st.write("מודל: Random Forest, ‏300 עצים, עומק מרבי 15")
    st.caption("החיזוי הוא כיוון יומי: עלייה (1) או ירידה/אי-עלייה (0).")

advisor_tab, comparison_tab, prediction_tab, new_data_tab, about_tab = st.tabs(
    [
        "🧭 התאמת השקעה ללקוח",
        "📊 השוואת מניות וטווחים",
        "🤖 תחזית וביצועי המודל",
        "➕ קליטת נתונים חדשים",
        "ℹ️ אודות ומתודולוגיה",
    ]
)


with advisor_tab:
    st.subheader("שאלון פרופיל והמלצת התאמה")
    st.write(
        "המערכת מדרגת את שלוש המניות לפי תשואה, תנודתיות, ירידה מרבית, יחס תשואה-סיכון, "
        "מומנטום והסתברות המודל."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_level = st.selectbox("רמת סיכון", ["נמוכה", "בינונית", "גבוהה"])
        goal = st.selectbox("מטרת ההשקעה", ["שמירה על ההון", "צמיחה", "מקסום תשואה"])
    with col2:
        horizon = st.selectbox("טווח השקעה", ["קצר (עד שנה)", "בינוני (1–3 שנים)", "ארוך (מעל 3 שנים)"])
        volatility_tolerance = st.selectbox("סבילות לתנודתיות", ["נמוכה", "בינונית", "גבוהה"])
    with col3:
        company_style = st.selectbox(
            "סגנון חברה מועדף",
            ["ללא העדפה", "חברה מבוססת ויציבה יחסית", "צמיחה אגרסיבית וחדשנות"],
        )
        default_advisor_start = max(min_project_date, date(2023, 1, 1))
        advisor_dates = st.date_input(
            "תקופת נתונים לחישוב ההתאמה",
            value=(default_advisor_start, max_project_date),
            min_value=min_project_date,
            max_value=max_project_date,
        )

    if isinstance(advisor_dates, tuple) and len(advisor_dates) == 2:
        advisor_start, advisor_end = advisor_dates
    else:
        advisor_start, advisor_end = default_advisor_start, max_project_date

    advisor_frame = project_data[
        project_data["date"].between(pd.Timestamp(advisor_start), pd.Timestamp(advisor_end))
    ].copy()
    advisor_metrics = calculate_stock_metrics(advisor_frame)

    if advisor_metrics.empty or len(advisor_metrics) < 2:
        st.error("אין מספיק נתונים בתקופה שנבחרה. בחרו טווח רחב יותר.")
    else:
        scored = score_recommendations(
            advisor_metrics,
            risk_level,
            horizon,
            goal,
            volatility_tolerance,
            company_style,
        )
        recommended_symbol = scored.index[0]
        recommended = scored.iloc[0]
        profile = {"risk": risk_level, "horizon": horizon, "goal": goal}

        st.markdown(
            f"""
            <div class="advisor-card">
                <h3>ההתאמה הגבוהה ביותר: {recommended_symbol} — {STOCK_LABELS.get(recommended_symbol, recommended_symbol)}</h3>
                <p>{build_recommendation_explanation(recommended_symbol, recommended, profile)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ציון התאמה", f"{recommended['fit_score']:.1%}")
        m2.metric("תשואה מצטברת בתקופה", f"{recommended['total_return']:.1%}")
        m3.metric("תנודתיות שנתית", f"{recommended['annualized_volatility']:.1%}")
        m4.metric("ירידה מרבית", f"{recommended['max_drawdown']:.1%}")

        display_scored = format_metrics_for_display(scored, include_score=True)
        st.dataframe(
            display_scored.style.format(metric_formats(include_score=True)),
            use_container_width=True,
            hide_index=True,
        )

        chart_data = advisor_frame[["date", "symbol", "adj_close"]].copy()
        chart_data["מחיר מנורמל"] = chart_data.groupby("symbol")["adj_close"].transform(
            lambda series: series / series.iloc[0] * 100
        )
        fig = px.line(
            chart_data,
            x="date",
            y="מחיר מנורמל",
            color="symbol",
            title="השוואת ביצועים מנורמלת — נקודת התחלה 100",
            labels={"date": "תאריך", "symbol": "מניה"},
        )
        st.plotly_chart(style_plotly(fig), use_container_width=True)
        st.caption(
            "השקלול הוא מנגנון עסקי שקוף שנבנה לצורך הפרויקט. הוא אינו מודל נוסף ואינו מבטיח תשואה עתידית."
        )


with comparison_tab:
    st.subheader("בחירת מניות, מקור נתונים וטווח תאריכים")
    c1, c2 = st.columns([1, 2])
    with c1:
        comparison_source = st.radio(
            "מקור הנתונים",
            ["נתוני הפרויקט", "Yahoo Finance — נתונים עדכניים"],
            help="הנתונים העדכניים משמשים להשוואת מחירים בלבד. חיזוי המודל מבוסס על מבנה נתוני הפרויקט.",
        )
        selected_symbols = st.multiselect(
            "מניות להשוואה",
            options=sorted(project_data["symbol"].unique()),
            default=sorted(project_data["symbol"].unique()),
        )

    with c2:
        if comparison_source == "נתוני הפרויקט":
            compare_dates = st.date_input(
                "טווח תאריכים",
                value=(max(min_project_date, date(2023, 1, 1)), max_project_date),
                min_value=min_project_date,
                max_value=max_project_date,
                key="project_compare_dates",
            )
        else:
            live_default_start = date.today() - timedelta(days=365)
            compare_dates = st.date_input(
                "טווח תאריכים",
                value=(live_default_start, date.today()),
                max_value=date.today(),
                key="live_compare_dates",
            )

    if not selected_symbols:
        st.info("בחרו לפחות מניה אחת.")
    else:
        if isinstance(compare_dates, tuple) and len(compare_dates) == 2:
            compare_start, compare_end = compare_dates
        else:
            compare_start = max_project_date - timedelta(days=365)
            compare_end = max_project_date

        if comparison_source == "נתוני הפרויקט":
            comparison_frame = project_data[
                project_data["symbol"].isin(selected_symbols)
                & project_data["date"].between(pd.Timestamp(compare_start), pd.Timestamp(compare_end))
            ].copy()
        else:
            with st.spinner("טוען נתוני שוק עדכניים..."):
                comparison_frame = fetch_live_prices(tuple(selected_symbols), compare_start, compare_end)
            if comparison_frame.empty:
                st.error("לא התקבלו נתונים מ-Yahoo Finance. נסו שוב או עברו לנתוני הפרויקט.")

        if not comparison_frame.empty:
            comparison_metrics = calculate_stock_metrics(comparison_frame)
            if not comparison_metrics.empty:
                st.dataframe(
                    format_metrics_for_display(comparison_metrics).style.format(metric_formats()),
                    use_container_width=True,
                    hide_index=True,
                )

            normalized = comparison_frame[["date", "symbol", "adj_close"]].dropna().copy()
            normalized["מחיר מנורמל"] = normalized.groupby("symbol")["adj_close"].transform(
                lambda series: series / series.iloc[0] * 100
            )
            price_fig = px.line(
                normalized,
                x="date",
                y="מחיר מנורמל",
                color="symbol",
                title="השוואת מחיר מנורמל",
                labels={"date": "תאריך", "symbol": "מניה"},
            )
            st.plotly_chart(style_plotly(price_fig), use_container_width=True)

            if not comparison_metrics.empty:
                risk_return = comparison_metrics.reset_index()
                scatter = px.scatter(
                    risk_return,
                    x="annualized_volatility",
                    y="annualized_return",
                    text="symbol",
                    size="observations",
                    title="השוואת סיכון מול תשואה",
                    labels={
                        "annualized_volatility": "תנודתיות שנתית",
                        "annualized_return": "תשואה שנתית מחושבת",
                        "observations": "תצפיות",
                    },
                )
                scatter.update_traces(textposition="top center")
                scatter.update_xaxes(tickformat=".0%")
                scatter.update_yaxes(tickformat=".0%")
                st.plotly_chart(style_plotly(scatter), use_container_width=True)


with prediction_tab:
    st.subheader("תחזיות ה-Random Forest ובדיקת ביצועים")
    st.info(
        "לבדיקה אמינה מומלץ לבחור את תקופת ה-Test בלבד. תחזיות מתקופת Train אינן מדד הוגן לביצועים על נתונים חדשים."
    )

    p1, p2 = st.columns(2)
    with p1:
        prediction_symbol = st.selectbox(
            "בחרו מניה",
            options=sorted(project_data["symbol"].unique()),
            key="prediction_symbol",
        )
    with p2:
        prediction_dates = st.date_input(
            "טווח להצגת התחזיות",
            value=(test_start_date, max_project_date),
            min_value=min_project_date,
            max_value=max_project_date,
            key="prediction_dates",
        )

    if isinstance(prediction_dates, tuple) and len(prediction_dates) == 2:
        prediction_start, prediction_end = prediction_dates
    else:
        prediction_start, prediction_end = test_start_date, max_project_date

    prediction_frame = project_data[
        (project_data["symbol"] == prediction_symbol)
        & project_data["date"].between(pd.Timestamp(prediction_start), pd.Timestamp(prediction_end))
    ].copy()

    if prediction_frame.empty:
        st.warning("אין נתונים למניה ולטווח שנבחרו.")
    else:
        selected_metrics = compute_classification_metrics(prediction_frame)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Accuracy", f"{selected_metrics['accuracy']:.2%}")
        k2.metric("Precision", f"{selected_metrics['precision']:.2%}")
        k3.metric("Recall", f"{selected_metrics['recall']:.2%}")
        k4.metric("F1", f"{selected_metrics['f1']:.2%}")
        k5.metric("AUC", "לא זמין" if pd.isna(selected_metrics["auc"]) else f"{selected_metrics['auc']:.2f}")

        probability_fig = go.Figure()
        probability_fig.add_trace(
            go.Scatter(
                x=prediction_frame["date"],
                y=prediction_frame["predicted_probability"],
                mode="lines",
                name="הסתברות חזויה לעלייה",
            )
        )
        probability_fig.add_hline(y=0.5, line_dash="dash", annotation_text="סף סיווג 0.5")
        probability_fig.update_layout(
            title=f"הסתברות חזויה לעלייה — {prediction_symbol}",
            xaxis_title="תאריך",
            yaxis_title="הסתברות",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(style_plotly(probability_fig), use_container_width=True)

        left, right = st.columns(2)
        with left:
            cm = confusion_matrix(
                prediction_frame["actual_direction"],
                prediction_frame["predicted_direction"],
                labels=[0, 1],
            )
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                x=["חזוי: ירידה", "חזוי: עלייה"],
                y=["בפועל: ירידה", "בפועל: עלייה"],
                title="Confusion Matrix",
                labels={"x": "תחזית", "y": "תוצאה בפועל", "color": "כמות"},
            )
            st.plotly_chart(style_plotly(cm_fig), use_container_width=True)
        with right:
            importance = pd.DataFrame(
                {
                    "מאפיין": [FEATURE_LABELS.get(name, name) for name in FEATURES],
                    "חשיבות": model.feature_importances_,
                }
            ).sort_values("חשיבות", ascending=True)
            importance_fig = px.bar(
                importance,
                x="חשיבות",
                y="מאפיין",
                orientation="h",
                title="חשיבות מאפיינים במודל",
            )
            st.plotly_chart(style_plotly(importance_fig), use_container_width=True)

        prediction_table = prediction_frame[
            [
                "date",
                "symbol",
                "dataset_split",
                "stock_return",
                "actual_direction",
                "predicted_direction",
                "predicted_probability",
            ]
        ].copy()
        prediction_table.columns = [
            "תאריך",
            "מניה",
            "קבוצת נתונים",
            "תשואה בפועל",
            "כיוון בפועל",
            "כיוון חזוי",
            "הסתברות לעלייה",
        ]
        prediction_table["כיוון בפועל"] = prediction_table["כיוון בפועל"].map({0: "ירידה", 1: "עלייה"})
        prediction_table["כיוון חזוי"] = prediction_table["כיוון חזוי"].map({0: "ירידה", 1: "עלייה"})
        st.dataframe(
            prediction_table.sort_values("תאריך", ascending=False).style.format(
                {"תשואה בפועל": "{:.2%}", "הסתברות לעלייה": "{:.1%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    test_frame = project_data[project_data["dataset_split"] == "Test"]
    overall_test_metrics = compute_classification_metrics(test_frame)
    st.caption(
        "ביצועי המודל על כלל קבוצת ה-Test: "
        f"Accuracy {overall_test_metrics['accuracy']:.2%}, "
        f"Precision {overall_test_metrics['precision']:.2%}, "
        f"Recall {overall_test_metrics['recall']:.2%}, "
        f"F1 {overall_test_metrics['f1']:.2%}, "
        f"AUC {overall_test_metrics['auc']:.2f}."
    )


with new_data_tab:
    st.subheader("קליטת רשומה חדשה וחיזוי כיוון")
    st.write(
        "בחרו רשומה קיימת כנקודת התחלה, עדכנו את הערכים ולחצו על חיזוי. "
        "כך המערכת מממשת קליטת נתונים חדשים ללא תלות בחיבור למסד הנתונים המקומי."
    )

    base_symbol = st.selectbox(
        "מניה לצורך ערכי ברירת מחדל",
        options=sorted(project_data["symbol"].unique()),
        key="new_data_symbol",
    )
    base_row = project_data[project_data["symbol"] == base_symbol].sort_values("date").iloc[-1]
    st.caption(f"ערכי ברירת המחדל נלקחו מהרשומה האחרונה הזמינה: {base_row['date']:%d/%m/%Y}")

    with st.form("manual_prediction_form"):
        a1, a2, a3 = st.columns(3)
        manual_values: dict[str, float] = {}
        for index, feature in enumerate(FEATURES):
            target_column = [a1, a2, a3][index % 3]
            with target_column:
                is_volume = feature == "volume"
                manual_values[feature] = st.number_input(
                    FEATURE_LABELS[feature],
                    min_value=0.0 if is_volume or feature in {"open_price", "adj_close", "SP500", "VIX", "US10Y"} else None,
                    value=float(base_row[feature]),
                    step=100000.0 if is_volume else 0.01,
                    format="%.0f" if is_volume else "%.4f",
                )
        submitted = st.form_submit_button("חשב תחזית")

    if submitted:
        new_frame = pd.DataFrame([manual_values], columns=FEATURES)
        new_prediction = int(model.predict(new_frame)[0])
        new_probability = float(model.predict_proba(new_frame)[0, 1])
        if new_prediction == 1:
            st.success(f"תחזית המודל: עלייה | הסתברות מחושבת לעלייה: {new_probability:.1%}")
        else:
            st.error(f"תחזית המודל: ירידה/אי-עלייה | הסתברות מחושבת לעלייה: {new_probability:.1%}")
        st.caption(
            "התחזית תקפה רק בהנחה שכל תשעת ערכי הקלט זמינים. המודל אינו מייצר בעצמו מחירי עתיד או נתוני מאקרו עתידיים."
        )

    st.divider()
    st.subheader("חיזוי מרובה באמצעות CSV")
    sample_rows = project_data.sort_values("date").groupby("symbol").tail(1)[FEATURES].copy()
    sample_csv = sample_rows.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "הורדת קובץ דוגמה",
        data=sample_csv,
        file_name="sample_new_data.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("העלו CSV הכולל את תשעת מאפייני המודל", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            missing = [column for column in FEATURES if column not in batch.columns]
            if missing:
                st.error("חסרות עמודות: " + ", ".join(missing))
            else:
                batch_features = batch[FEATURES].apply(pd.to_numeric, errors="coerce")
                if batch_features.isna().any().any():
                    st.error("אחד או יותר מערכי הקלט אינם מספריים או חסרים.")
                else:
                    output = batch.copy()
                    output["Predicted_Direction"] = model.predict(batch_features)
                    output["Predicted_Label"] = output["Predicted_Direction"].map({0: "DOWN", 1: "UP"})
                    output["Predicted_Probability"] = model.predict_proba(batch_features)[:, 1]
                    st.dataframe(output, use_container_width=True, hide_index=True)
                    st.download_button(
                        "הורדת תוצאות החיזוי",
                        data=output.to_csv(index=False).encode("utf-8-sig"),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as exc:
            st.error(f"לא ניתן לקרוא את הקובץ: {exc}")


with about_tab:
    st.subheader("איך המערכת עונה על דרישות שלב ה-Deployment")
    st.markdown(
        """
        **קליטת נתונים חדשים:** טופס ידני והעלאת CSV לחיזוי מרובה.  
        **הצגת ביצועי המודל:** Accuracy, Precision, Recall, F1, AUC ו-Confusion Matrix.  
        **ויזואליזציות:** השוואת מחירים, תשואה מול סיכון, הסתברות חזויה וחשיבות מאפיינים.  
        **בחירת מניה וטווח:** בכלי ההשוואה ובמסך התחזית.  
        **נתונים עדכניים:** אפשרות למשיכת מחירי שוק מ-Yahoo Finance לצורך השוואה.  
        **שימוש עסקי:** שאלון לקוח ומנגנון דירוג שקוף בין שלוש חלופות.
        """
    )

    st.subheader("מקורות ומבנה הנתונים")
    st.write(
        "מחירי המניות נאספו עבור AAPL, GOOGL ו-NVDA, ונוספו משתני מאקרו: ריבית, אינפלציה, "
        "אמון צרכנים, S&P 500, VIX ותשואת אג״ח ל-10 שנים. החיבור נעשה לפי תאריך."
    )
