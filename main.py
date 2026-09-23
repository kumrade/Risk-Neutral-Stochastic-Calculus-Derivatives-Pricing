"""
HELM Risk-Neutral Stochastic Calculus & Derivatives Pricing Laboratory
=====================================================================

A single-file educational / portfolio research application that connects:

Historical Prices -> Returns -> Brownian Motion -> GBM -> Ito's Lemma
-> Physical Measure P -> Risk-Neutral Measure Q -> Monte Carlo Pricing
-> Black-Scholes & Greeks -> Dynamic Delta Hedging -> Diagnostics

The program intentionally separates:
    * historical estimation under the physical measure P, and
    * arbitrage-free derivative valuation under the risk-neutral measure Q.

Historical data input
---------------------
The primary data path is Angel One SmartAPI: log in, resolve an NSE equity, fetch
completed ONE_DAY candles, and use the latest LTP as the current spot S0. A CSV
loader and synthetic demo history remain available as offline fallbacks.

Install market-data dependencies:
    pip install smartapi-python pyotp logzero websocket-client pycryptodome

Run:
    python HELM_Risk_Neutral_Stochastic_Calculus_Lab_SMARTAPI.py

Self-test:
    python HELM_Risk_Neutral_Stochastic_Calculus_Lab_SMARTAPI.py --self-test

Interactive real-price SmartAPI test:
    python HELM_Risk_Neutral_Stochastic_Calculus_Lab_SMARTAPI.py --smartapi-smoke-test
"""

from __future__ import annotations

import getpass
import json
import math
import os
import sys
import tempfile
import threading
import time
import urllib.request
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from scipy.stats import norm, shapiro, jarque_bera

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRADING_DAYS = 252
EPS = 1e-12


# =============================================================================
# SMARTAPI MARKET-DATA CONNECTION
# =============================================================================

# SmartAPI is deliberately imported only when the user clicks Connect.  The
# mathematical laboratories and --self-test therefore remain usable even on a
# machine where the broker SDK is not installed yet.

INSTRUMENT_MASTER_URLS = (
    "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json",
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json",
)

EMERGENCY_NSE_CASH_MAP: dict[str, tuple[str, str]] = {
    "RELIANCE": ("RELIANCE-EQ", "2885"),
    "SBIN": ("SBIN-EQ", "3045"),
    "TCS": ("TCS-EQ", "11536"),
    "INFY": ("INFY-EQ", "1594"),
    "HDFCBANK": ("HDFCBANK-EQ", "1333"),
    "ITC": ("ITC-EQ", "1660"),
    "ICICIBANK": ("ICICIBANK-EQ", "4963"),
    "AXISBANK": ("AXISBANK-EQ", "5900"),
}


def _select_writable_market_data_directory() -> Path:
    candidates: list[Path] = []
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if root:
            candidates.append(Path(root) / "HELM" / "RiskNeutralLab")
    candidates.extend([
        Path.home() / ".helm_risk_neutral_lab",
        Path(tempfile.gettempdir()) / "HELM_RiskNeutralLab",
    ])
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate.resolve()
        except Exception:
            continue
    raise RuntimeError("No writable application-data folder is available for SmartAPI logs/cache.")


APP_DATA_DIR = _select_writable_market_data_directory()
SMARTAPI_RUNTIME_DIR = APP_DATA_DIR / "smartapi_runtime"
SMARTAPI_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
_CWD_LOCK = threading.RLock()


@contextmanager
def temporary_working_directory(directory: Path):
    """Give SmartAPI a writable CWD for releases that create ./logs."""
    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with _CWD_LOCK:
        old = Path.cwd()
        os.chdir(target)
        try:
            yield target
        finally:
            os.chdir(old)


@dataclass
class SmartAPISession:
    api: object
    api_key: str
    client_code: str
    auth_token: str
    refresh_token: str
    feed_token: str


class SmartAPIMarketData:
    """Small, read-only SmartAPI gateway for equity prices used by this lab."""

    def __init__(self) -> None:
        self.session: Optional[SmartAPISession] = None
        self.instrument_master_cache: Optional[list[dict[str, object]]] = None
        self.instrument_master_file = APP_DATA_DIR / "OpenAPIScripMaster.json"
        self.symbol_cache: dict[tuple[str, str], tuple[str, str]] = {}
        self.history_cache: dict[tuple, pd.DataFrame] = {}

    @staticmethod
    def _imports():
        try:
            from SmartApi import SmartConnect
            import pyotp
        except ImportError as exc:
            raise RuntimeError(
                "SmartAPI packages are missing. Install:\n"
                "pip install smartapi-python pyotp logzero websocket-client pycryptodome"
            ) from exc
        return SmartConnect, pyotp

    @staticmethod
    def _construct(factory: Callable, *args, **kwargs):
        try:
            with temporary_working_directory(SMARTAPI_RUNTIME_DIR):
                return factory(*args, **kwargs)
        except PermissionError as exc:
            raise RuntimeError(
                "SmartAPI could not create its log files inside the HELM user-data folder: "
                f"{SMARTAPI_RUNTIME_DIR}"
            ) from exc

    def login(self, api_key: str, client_code: str, pin: str, totp_secret: str) -> SmartAPISession:
        SmartConnect, pyotp = self._imports()
        api_key = api_key.strip()
        client_code = client_code.strip().upper()
        pin = pin.strip()
        secret = totp_secret.replace(" ", "").strip()
        if not all((api_key, client_code, pin, secret)):
            raise ValueError("API key, client code, PIN/MPIN and TOTP secret are required.")
        try:
            totp = pyotp.TOTP(secret).now()
        except Exception as exc:
            raise RuntimeError("The supplied TOTP secret is invalid.") from exc

        api = self._construct(SmartConnect, api_key=api_key)
        response = api.generateSession(client_code, pin, totp)
        if not isinstance(response, dict) or not response.get("status"):
            raise RuntimeError(f"SmartAPI login failed: {response}")
        data = response.get("data") or {}
        auth_token = str(data.get("jwtToken") or "")
        refresh_token = str(data.get("refreshToken") or "")
        feed_token = str(api.getfeedToken() or data.get("feedToken") or "")
        if not auth_token or not refresh_token:
            raise RuntimeError("SmartAPI login succeeded but did not return complete session tokens.")
        self.session = SmartAPISession(api, api_key, client_code, auth_token, refresh_token, feed_token)
        self.history_cache.clear()
        return self.session

    def require_session(self) -> SmartAPISession:
        if self.session is None:
            raise RuntimeError("Connect to SmartAPI first.")
        return self.session

    def logout(self) -> None:
        session = self.session
        self.session = None
        self.history_cache.clear()
        if session is not None:
            try:
                session.api.terminateSession(session.client_code)
            except Exception:
                pass

    @staticmethod
    def normalize_symbol(query: str) -> str:
        raw = str(query or "").strip().upper()
        if ":" in raw:
            raw = raw.split(":", 1)[1]
        if raw.endswith(".NS"):
            raw = raw[:-3]
        if raw.endswith("-EQ"):
            raw = raw[:-3]
        return raw.strip()

    @staticmethod
    def _candidate_symbol(row: dict[str, object]) -> str:
        for key in ("tradingsymbol", "tradeSymbol", "symbol", "tradingSymbol"):
            value = row.get(key)
            if value:
                return str(value).strip().upper()
        return ""

    @staticmethod
    def _candidate_token(row: dict[str, object], exchange: str) -> str:
        exchange = exchange.upper()
        if exchange == "NSE":
            for key in ("nseCashToken", "symboltoken", "symbolToken", "token"):
                value = row.get(key)
                if value not in (None, "", 0, "0"):
                    return str(value).strip()
        for key in ("symboltoken", "symbolToken", "token"):
            value = row.get(key)
            if value not in (None, "", 0, "0"):
                return str(value).strip()
        return ""

    @staticmethod
    def _extract_search_rows(response: object) -> list[dict[str, object]]:
        if not isinstance(response, dict):
            return []
        data = response.get("data")
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("dataList", "results", "result"):
                rows = data.get(key)
                if isinstance(rows, list):
                    return [x for x in rows if isinstance(x, dict)]
        return []

    def _download_instrument_master(self) -> list[dict[str, object]]:
        last_error: Optional[Exception] = None
        for url in INSTRUMENT_MASTER_URLS:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    decoded = json.loads(response.read().decode("utf-8-sig"))
                if not isinstance(decoded, list) or not decoded:
                    raise RuntimeError("Instrument master response was empty or malformed.")
                tmp = self.instrument_master_file.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(decoded), encoding="utf-8")
                tmp.replace(self.instrument_master_file)
                return decoded
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Unable to download the Angel One instrument master: {last_error}")

    def _load_instrument_master(self) -> list[dict[str, object]]:
        if self.instrument_master_cache is not None:
            return self.instrument_master_cache
        # Reuse a reasonably recent cache first.
        if self.instrument_master_file.exists():
            try:
                age = datetime.now() - datetime.fromtimestamp(self.instrument_master_file.stat().st_mtime)
                if age < timedelta(hours=18):
                    decoded = json.loads(self.instrument_master_file.read_text(encoding="utf-8-sig"))
                    if isinstance(decoded, list) and decoded:
                        self.instrument_master_cache = decoded
                        return decoded
            except Exception:
                pass
        try:
            decoded = self._download_instrument_master()
            self.instrument_master_cache = decoded
            return decoded
        except Exception:
            if self.instrument_master_file.exists():
                decoded = json.loads(self.instrument_master_file.read_text(encoding="utf-8-sig"))
                if isinstance(decoded, list) and decoded:
                    self.instrument_master_cache = decoded
                    return decoded
            raise

    def _master_match(self, exchange: str, base: str) -> Optional[tuple[str, str]]:
        exchange = exchange.upper()
        target = f"{base}-EQ" if exchange == "NSE" else base
        master = self._load_instrument_master()
        exact: list[tuple[str, str]] = []
        loose: list[tuple[str, str]] = []
        for row in master:
            exch = str(row.get("exch_seg") or row.get("exchange") or "").upper()
            if exch and exch != exchange:
                continue
            symbol = self._candidate_symbol(row)
            token = self._candidate_token(row, exchange)
            if not symbol or not token:
                continue
            inst = str(row.get("instrumenttype") or row.get("instrumentType") or "").upper()
            if symbol == target:
                exact.append((symbol, token))
            elif symbol.startswith(base) and (exchange != "NSE" or symbol.endswith("-EQ") or inst in {"EQ", ""}):
                loose.append((symbol, token))
        if exact:
            return exact[0]
        if loose:
            return loose[0]
        return None

    def resolve_symbol(self, exchange: str, query: str, token_override: str = "") -> tuple[str, str]:
        session = self.require_session()
        exchange = exchange.upper().strip()
        base = self.normalize_symbol(query)
        if not base:
            raise ValueError("Enter a stock symbol, for example INFY or SBIN.")
        cache_key = (exchange, base)
        if token_override.strip():
            symbol = f"{base}-EQ" if exchange == "NSE" else base
            result = (symbol, token_override.strip())
            self.symbol_cache[cache_key] = result
            return result
        if cache_key in self.symbol_cache:
            return self.symbol_cache[cache_key]

        selected: Optional[tuple[str, str]] = None
        try:
            selected = self._master_match(exchange, base)
        except Exception:
            selected = None

        if selected is None:
            search_terms = [base, f"{base}-EQ"] if exchange == "NSE" else [base]
            rows: list[dict[str, object]] = []
            for term in search_terms:
                try:
                    response = session.api.searchScrip(exchange, term)
                    rows.extend(self._extract_search_rows(response))
                except Exception:
                    continue
            target = f"{base}-EQ" if exchange == "NSE" else base
            for row in rows:
                symbol = self._candidate_symbol(row)
                token = self._candidate_token(row, exchange)
                if symbol == target and token:
                    selected = (symbol, token)
                    break
            if selected is None:
                for row in rows:
                    symbol = self._candidate_symbol(row)
                    token = self._candidate_token(row, exchange)
                    if symbol.startswith(base) and token:
                        selected = (symbol, token)
                        break

        if selected is None and exchange == "NSE" and base in EMERGENCY_NSE_CASH_MAP:
            selected = EMERGENCY_NSE_CASH_MAP[base]
        if selected is None:
            raise RuntimeError(
                f"Could not resolve {exchange}:{query}. Try the base NSE symbol (e.g. INFY), "
                "or paste the Angel One symbol token in Token override."
            )
        self.symbol_cache[cache_key] = selected
        return selected

    def ltp_snapshot(self, exchange: str, trading_symbol: str, token: str) -> dict[str, object]:
        session = self.require_session()
        response = session.api.ltpData(exchange.upper(), trading_symbol, str(token))
        if not isinstance(response, dict) or not response.get("status"):
            raise RuntimeError(f"LTP request failed: {response}")
        return response.get("data") or {}

    @staticmethod
    def _date_chunks(start: datetime, end: datetime, days: int = 365):
        # SmartAPI ONE_DAY requests are split on full trading-day boundaries so
        # no daily candle is accidentally skipped at a chunk edge.
        cursor = start.replace(hour=9, minute=15, second=0, microsecond=0)
        final = end.replace(second=0, microsecond=0)
        while cursor < final:
            candidate_date = min((cursor + timedelta(days=days)).date(), final.date())
            chunk_end = datetime.combine(candidate_date, datetime.min.time()).replace(hour=15, minute=30)
            chunk_end = min(chunk_end, final)
            yield cursor, chunk_end
            cursor = (chunk_end + timedelta(days=1)).replace(hour=9, minute=15, second=0, microsecond=0)

    def historical_candles(
        self,
        exchange: str,
        token: str,
        start: datetime,
        end: datetime,
        progress: Optional[Callable[[str], None]] = None,
    ) -> pd.DataFrame:
        session = self.require_session()
        if start >= end:
            raise ValueError("Historical start date must be before end date.")
        cache_key = (exchange.upper(), str(token), start.isoformat(), end.isoformat())
        if cache_key in self.history_cache:
            return self.history_cache[cache_key].copy()

        frames: list[pd.DataFrame] = []
        chunks = list(self._date_chunks(start, end, 365))
        for number, (chunk_start, chunk_end) in enumerate(chunks, 1):
            if progress:
                progress(f"Daily candles {number}/{len(chunks)}: {chunk_start:%Y-%m-%d} → {chunk_end:%Y-%m-%d}")
            params = {
                "exchange": exchange.upper(),
                "symboltoken": str(token),
                "interval": "ONE_DAY",
                "fromdate": chunk_start.strftime("%Y-%m-%d %H:%M"),
                "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
            }
            last_error: object = None
            for attempt in range(3):
                try:
                    response = session.api.getCandleData(params)
                    if isinstance(response, dict) and response.get("status"):
                        rows = response.get("data") or []
                        if rows:
                            frames.append(pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"]))
                        last_error = None
                        break
                    last_error = response
                except Exception as exc:
                    last_error = exc
                time.sleep(1.0 + attempt)
            if last_error is not None:
                raise RuntimeError(f"Historical candle request failed: {last_error}")
            if number < len(chunks):
                time.sleep(0.6)

        if not frames:
            raise RuntimeError("SmartAPI returned no historical daily candles for this symbol/date range.")
        data = pd.concat(frames, ignore_index=True)
        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
        data = data.dropna(subset=["datetime"]).drop_duplicates("datetime")
        for column in ["open", "high", "low", "close", "volume"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=["close"]).set_index("datetime").sort_index()
        # Keep a timezone-naive trading-date index inside the mathematical lab.
        try:
            if getattr(data.index, "tz", None) is not None:
                data.index = data.index.tz_localize(None)
        except Exception:
            pass
        self.history_cache[cache_key] = data.copy()
        return data

    def fetch_equity_history(
        self,
        symbol: str,
        exchange: str,
        start: datetime,
        end: datetime,
        token_override: str = "",
        progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[pd.DataFrame, str, str, dict[str, object]]:
        trading_symbol, token = self.resolve_symbol(exchange, symbol, token_override)
        candles = self.historical_candles(exchange, token, start, end, progress)
        history = prepare_history(pd.DataFrame({"Close": candles["close"]}, index=candles.index))
        try:
            snapshot = self.ltp_snapshot(exchange, trading_symbol, token)
        except Exception:
            snapshot = {}
        return history, trading_symbol, token, snapshot


# =============================================================================
# HISTORICAL DATA AND RETURN ESTIMATION
# =============================================================================


def _detect_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_historical_csv(path: str | Path) -> pd.DataFrame:
    """Load and normalize a historical equity CSV into columns Date and Close."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(path)
    date_col = _detect_column(raw.columns, ["date", "datetime", "timestamp"])
    close_col = _detect_column(raw.columns, ["close", "adj close", "adjusted close", "ltp"])

    if date_col is None:
        raise ValueError("CSV must contain a Date/DATE/date style column.")
    if close_col is None:
        raise ValueError("CSV must contain a Close/CLOSE/close/Adj Close style column.")

    out = pd.DataFrame({
        "Date": pd.to_datetime(raw[date_col], errors="coerce"),
        "Close": pd.to_numeric(raw[close_col], errors="coerce"),
    }).dropna()
    out = out[out["Close"] > 0].sort_values("Date").drop_duplicates("Date")
    out = out.set_index("Date")
    if len(out) < 80:
        raise ValueError("Need at least 80 clean historical observations.")
    return prepare_history(out)


def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    """Add arithmetic and log returns to a clean Close-price series."""
    out = df.copy()
    if "Close" not in out.columns:
        if "close" in out.columns:
            out = out.rename(columns={"close": "Close"})
        else:
            raise ValueError("DataFrame must contain a Close column.")
    out = out.sort_index()
    out["Return"] = out["Close"].pct_change()
    out["Log_Return"] = np.log(out["Close"]).diff()
    return out


def make_demo_history(
    years: int = 5,
    s0: float = 1000.0,
    mu: float = 0.12,
    sigma: float = 0.24,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic-looking positive historical price series for demo use."""
    rng = np.random.default_rng(seed)
    n = int(years * TRADING_DAYS)
    dt = 1.0 / TRADING_DAYS
    z = rng.standard_normal(n)
    log_steps = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    prices = s0 * np.exp(np.cumsum(np.r_[0.0, log_steps]))
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 1)
    return prepare_history(pd.DataFrame({"Close": prices}, index=dates))


def historical_statistics(history: pd.DataFrame) -> pd.Series:
    clean = history["Log_Return"].dropna().astype(float)
    simple = history["Return"].dropna().astype(float)
    if len(clean) < 30:
        raise ValueError("Too few returns for historical statistics.")

    mu_log_daily = clean.mean()
    sigma_daily = clean.std(ddof=1)
    mu_log_annual = mu_log_daily * TRADING_DAYS
    sigma_annual = sigma_daily * math.sqrt(TRADING_DAYS)
    # Approximate arithmetic drift implied by log-return drift.
    mu_arith_annual = mu_log_annual + 0.5 * sigma_annual**2

    return pd.Series({
        "Observations": len(history),
        "Return observations": len(clean),
        "Last close": history["Close"].iloc[-1],
        "Daily arithmetic mean": simple.mean(),
        "Daily log-return mean": mu_log_daily,
        "Daily volatility": sigma_daily,
        "Annualized physical drift (mu)": mu_arith_annual,
        "Annualized volatility (sigma)": sigma_annual,
        "Annualized log drift": mu_log_annual,
    })


# =============================================================================
# BROWNIAN MOTION LABORATORY
# =============================================================================


def brownian_motion_lab(history: pd.DataFrame) -> dict[str, object]:
    """
    Build an empirical Brownian-motion approximation from standardized historical
    log-return shocks and measure the main Brownian properties numerically.
    """
    r = history["Log_Return"].dropna().astype(float)
    if len(r) < 50:
        raise ValueError("Need at least 50 returns for Brownian diagnostics.")

    dt = 1.0 / TRADING_DAYS
    standardized = (r - r.mean()) / max(r.std(ddof=1), EPS)
    dW = standardized * math.sqrt(dt)
    W = dW.cumsum()
    W = W - W.iloc[0]

    inc = dW.values
    autocorr = np.corrcoef(inc[:-1], inc[1:])[0, 1] if len(inc) > 2 else np.nan
    sample_for_shapiro = inc[-5000:] if len(inc) > 5000 else inc
    sh_stat, sh_p = shapiro(sample_for_shapiro) if len(sample_for_shapiro) >= 3 else (np.nan, np.nan)
    qv = float(np.sum(inc**2))
    theoretical_qv = len(inc) * dt

    return {
        "W": pd.Series(W.values, index=r.index, name="Empirical_W"),
        "dW": pd.Series(dW.values, index=r.index, name="dW"),
        "diagnostics": pd.Series({
            "Increment mean": float(np.mean(inc)),
            "Increment variance": float(np.var(inc, ddof=1)),
            "Theoretical increment variance dt": dt,
            "Lag-1 increment autocorrelation": float(autocorr),
            "Shapiro-Wilk p-value": float(sh_p),
            "Empirical quadratic variation": qv,
            "Theoretical quadratic variation": theoretical_qv,
            "QV ratio empirical/theoretical": qv / theoretical_qv if theoretical_qv > 0 else np.nan,
        }),
    }


# =============================================================================
# GEOMETRIC BROWNIAN MOTION
# =============================================================================


def simulate_gbm_paths(
    s0: float,
    drift: float,
    sigma: float,
    years: float,
    steps: int,
    paths: int,
    seed: int = 123,
    antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact GBM simulation: S_{t+dt} = S_t exp((drift-.5 sigma^2)dt + sigma sqrt(dt) Z)."""
    if s0 <= 0 or sigma < 0 or years <= 0 or steps < 1 or paths < 1:
        raise ValueError("Invalid GBM parameters.")
    rng = np.random.default_rng(seed)
    dt = years / steps

    if antithetic:
        half = int(math.ceil(paths / 2))
        z_half = rng.standard_normal((steps, half))
        z = np.concatenate([z_half, -z_half], axis=1)[:, :paths]
    else:
        z = rng.standard_normal((steps, paths))

    increments = (drift - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.vstack([np.zeros(paths), np.cumsum(increments, axis=0)])
    values = s0 * np.exp(log_paths)
    time = np.linspace(0.0, years, steps + 1)
    return time, values


def gbm_summary(paths: np.ndarray) -> pd.Series:
    terminal = paths[-1]
    return pd.Series({
        "Terminal mean": terminal.mean(),
        "Terminal median": np.median(terminal),
        "Terminal std": terminal.std(ddof=1),
        "5% terminal quantile": np.quantile(terminal, 0.05),
        "95% terminal quantile": np.quantile(terminal, 0.95),
        "Minimum simulated terminal": terminal.min(),
        "Maximum simulated terminal": terminal.max(),
    })


# =============================================================================
# ITO'S LEMMA LABORATORY
# =============================================================================


@dataclass
class ItoFunction:
    name: str
    f: Callable[[np.ndarray], np.ndarray]
    fp: Callable[[np.ndarray], np.ndarray]
    fpp: Callable[[np.ndarray], np.ndarray]


ITO_FUNCTIONS = {
    "ln(S)": ItoFunction(
        "ln(S)",
        lambda s: np.log(s),
        lambda s: 1.0 / s,
        lambda s: -1.0 / (s**2),
    ),
    "S^2": ItoFunction(
        "S^2",
        lambda s: s**2,
        lambda s: 2.0 * s,
        lambda s: np.full_like(s, 2.0),
    ),
    "sqrt(S)": ItoFunction(
        "sqrt(S)",
        lambda s: np.sqrt(s),
        lambda s: 0.5 / np.sqrt(s),
        lambda s: -0.25 / (s ** 1.5),
    ),
}


def ito_lemma_lab(
    s0: float,
    mu: float,
    sigma: float,
    years: float = 1.0,
    steps: int = TRADING_DAYS,
    function_name: str = "ln(S)",
    seed: int = 7,
) -> dict[str, object]:
    """Numerically compare ordinary first-order calculus with Ito's second-order correction."""
    if function_name not in ITO_FUNCTIONS:
        raise ValueError(f"Unknown Ito function: {function_name}")
    fn = ITO_FUNCTIONS[function_name]

    t, path_matrix = simulate_gbm_paths(s0, mu, sigma, years, steps, 1, seed=seed)
    s = path_matrix[:, 0]
    dt = years / steps
    s_prev = s[:-1]
    ds = np.diff(s)
    actual_df = np.diff(fn.f(s))

    ordinary = fn.fp(s_prev) * ds
    ito_correction = 0.5 * fn.fpp(s_prev) * (sigma**2) * (s_prev**2) * dt
    ito_approx = ordinary + ito_correction

    ordinary_error = actual_df - ordinary
    ito_error = actual_df - ito_approx

    frame = pd.DataFrame({
        "S": s_prev,
        "dS": ds,
        "Actual dF": actual_df,
        "Ordinary approx": ordinary,
        "Ito correction": ito_correction,
        "Ito approx": ito_approx,
        "Ordinary error": ordinary_error,
        "Ito error": ito_error,
    }, index=pd.RangeIndex(1, steps + 1, name="Step"))

    return {
        "time": t,
        "price_path": s,
        "frame": frame,
        "summary": pd.Series({
            "Function": function_name,
            "Ordinary calculus RMSE": math.sqrt(float(np.mean(ordinary_error**2))),
            "Ito approximation RMSE": math.sqrt(float(np.mean(ito_error**2))),
            "Mean absolute Ito correction": float(np.mean(np.abs(ito_correction))),
            "Final actual F(S)": float(fn.f(np.array([s[-1]]))[0]),
        }),
    }


# =============================================================================
# PHYSICAL MEASURE P VS RISK-NEUTRAL MEASURE Q
# =============================================================================


def p_vs_q_lab(
    s0: float,
    mu: float,
    r: float,
    q: float,
    sigma: float,
    years: float = 1.0,
    paths: int = 50_000,
    seed: int = 99,
) -> dict[str, object]:
    """Compare terminal distributions under P and Q and test the Q-martingale condition."""
    steps = max(1, int(round(TRADING_DAYS * years)))
    _, p_paths = simulate_gbm_paths(s0, mu, sigma, years, steps, paths, seed=seed, antithetic=True)
    _, q_paths = simulate_gbm_paths(s0, r - q, sigma, years, steps, paths, seed=seed + 1, antithetic=True)

    p_terminal = p_paths[-1]
    q_terminal = q_paths[-1]
    # With continuous dividend yield q, e^{-(r-q)t} S_t is a martingale under Q.
    discounted_q = math.exp(-(r - q) * years) * q_terminal
    martingale_mean = float(np.mean(discounted_q))

    summary = pd.Series({
        "Physical drift mu": mu,
        "Risk-neutral drift r-q": r - q,
        "P terminal mean": float(np.mean(p_terminal)),
        "Q terminal mean": float(np.mean(q_terminal)),
        "Discounted Q terminal mean": martingale_mean,
        "Initial spot": s0,
        "Martingale absolute error": martingale_mean - s0,
        "Martingale relative error": martingale_mean / s0 - 1.0,
    })
    return {"p_terminal": p_terminal, "q_terminal": q_terminal, "summary": summary}


# =============================================================================
# BLACK-SCHOLES, GREEKS, MONTE CARLO PRICING
# =============================================================================


def _validate_option_inputs(s0, k, t, r, q, sigma):
    if s0 <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        raise ValueError("Spot, strike, maturity and volatility must be positive.")


def d1_d2(s0: float, k: float, t: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    _validate_option_inputs(s0, k, t, r, q, sigma)
    d1 = (math.log(s0 / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return d1, d2


def black_scholes_price(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "Call",
) -> float:
    d1, d2 = d1_d2(s0, k, t, r, q, sigma)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)
    if option_type.lower().startswith("c"):
        return s0 * disc_q * norm.cdf(d1) - k * disc_r * norm.cdf(d2)
    return k * disc_r * norm.cdf(-d2) - s0 * disc_q * norm.cdf(-d1)


def black_scholes_greeks(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "Call",
) -> pd.Series:
    d1, d2 = d1_d2(s0, k, t, r, q, sigma)
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    pdf = norm.pdf(d1)

    is_call = option_type.lower().startswith("c")
    delta = disc_q * norm.cdf(d1) if is_call else disc_q * (norm.cdf(d1) - 1.0)
    gamma = disc_q * pdf / (s0 * sigma * math.sqrt(t))
    vega = s0 * disc_q * pdf * math.sqrt(t) / 100.0  # per 1 vol point

    common_theta = -(s0 * disc_q * pdf * sigma) / (2.0 * math.sqrt(t))
    if is_call:
        theta = (common_theta - r * k * disc_r * norm.cdf(d2) + q * s0 * disc_q * norm.cdf(d1)) / TRADING_DAYS
        rho = k * t * disc_r * norm.cdf(d2) / 100.0
    else:
        theta = (common_theta + r * k * disc_r * norm.cdf(-d2) - q * s0 * disc_q * norm.cdf(-d1)) / TRADING_DAYS
        rho = -k * t * disc_r * norm.cdf(-d2) / 100.0

    price = black_scholes_price(s0, k, t, r, q, sigma, option_type)
    return pd.Series({
        "Price": price,
        "Delta": delta,
        "Gamma": gamma,
        "Vega (per 1 vol point)": vega,
        "Theta (per trading day)": theta,
        "Rho (per 1 rate point)": rho,
        "d1": d1,
        "d2": d2,
    })


def risk_neutral_monte_carlo(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "Call",
    paths: int = 100_000,
    seed: int = 1234,
    antithetic: bool = True,
) -> dict[str, object]:
    _validate_option_inputs(s0, k, t, r, q, sigma)
    if paths < 100:
        raise ValueError("Use at least 100 Monte Carlo paths.")

    rng = np.random.default_rng(seed)
    if antithetic:
        half = int(math.ceil(paths / 2))
        z_half = rng.standard_normal(half)
        z = np.concatenate([z_half, -z_half])[:paths]
    else:
        z = rng.standard_normal(paths)

    st = s0 * np.exp((r - q - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z)
    if option_type.lower().startswith("c"):
        payoff = np.maximum(st - k, 0.0)
    else:
        payoff = np.maximum(k - st, 0.0)

    discounted = math.exp(-r * t) * payoff
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / math.sqrt(len(discounted)))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr
    bs = black_scholes_price(s0, k, t, r, q, sigma, option_type)

    return {
        "terminal": st,
        "discounted_payoff": discounted,
        "summary": pd.Series({
            "Monte Carlo price": price,
            "Black-Scholes price": bs,
            "MC - BS difference": price - bs,
            "Monte Carlo standard error": stderr,
            "95% CI low": ci_low,
            "95% CI high": ci_high,
            "Paths": paths,
        }),
    }


def monte_carlo_convergence(
    s0: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
    checkpoints: tuple[int, ...] = (500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000),
    seed: int = 77,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    max_n = max(checkpoints)
    half = int(math.ceil(max_n / 2))
    z_half = rng.standard_normal(half)
    z = np.concatenate([z_half, -z_half])[:max_n]
    st = s0 * np.exp((r - q - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z)
    payoff = np.maximum(st - k, 0.0) if option_type.lower().startswith("c") else np.maximum(k - st, 0.0)
    discounted = math.exp(-r * t) * payoff
    bs = black_scholes_price(s0, k, t, r, q, sigma, option_type)

    rows = []
    for n in checkpoints:
        sample = discounted[:n]
        mean = float(sample.mean())
        se = float(sample.std(ddof=1) / math.sqrt(n))
        rows.append({"Paths": n, "MC Price": mean, "Std Error": se, "BS Price": bs, "Abs Error": abs(mean - bs)})
    return pd.DataFrame(rows)


# =============================================================================
# DYNAMIC DELTA HEDGING
# =============================================================================


def _bs_delta(s, k, t, r, q, sigma, option_type="Call"):
    if t <= 0:
        if option_type.lower().startswith("c"):
            return 1.0 if s > k else 0.0
        return -1.0 if s < k else 0.0
    d1, _ = d1_d2(float(s), k, t, r, q, sigma)
    if option_type.lower().startswith("c"):
        return math.exp(-q * t) * norm.cdf(d1)
    return math.exp(-q * t) * (norm.cdf(d1) - 1.0)


def delta_hedging_backtest(
    s0: float,
    k: float,
    t: float,
    mu: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "Call",
    paths: int = 1_000,
    rebalance_steps: int = 63,
    transaction_cost_bps: float = 0.0,
    seed: int = 2026,
) -> dict[str, object]:
    """
    Simulate realized stock paths under P (drift=mu), while pricing and hedging the
    option with Black-Scholes under Q. Returns terminal hedging-error distribution.
    """
    _validate_option_inputs(s0, k, t, r, q, sigma)
    if paths < 1 or rebalance_steps < 1:
        raise ValueError("paths and rebalance_steps must be positive.")

    dt = t / rebalance_steps
    _, s_paths = simulate_gbm_paths(s0, mu, sigma, t, rebalance_steps, paths, seed=seed, antithetic=True)
    premium = black_scholes_price(s0, k, t, r, q, sigma, option_type)

    errors = np.zeros(paths)
    total_costs = np.zeros(paths)
    sample_records = []

    for j in range(paths):
        s = s_paths[:, j]
        delta = _bs_delta(s[0], k, t, r, q, sigma, option_type)
        shares = delta
        cash = premium - shares * s[0]
        path_cost = 0.0

        if j == 0:
            sample_records.append({"Step": 0, "Spot": s[0], "Delta": delta, "Shares": shares, "Cash": cash, "Option": premium})

        for i in range(1, rebalance_steps + 1):
            # Financing and dividend cash flow over previous interval.
            cash *= math.exp(r * dt)
            if q != 0:
                cash += shares * s[i - 1] * (math.exp(q * dt) - 1.0)

            if i < rebalance_steps:
                remaining = t - i * dt
                new_delta = _bs_delta(s[i], k, remaining, r, q, sigma, option_type)
                trade = new_delta - shares
                tc = abs(trade) * s[i] * transaction_cost_bps / 10_000.0
                cash -= trade * s[i] + tc
                path_cost += tc
                shares = new_delta
                if j == 0:
                    option_value = black_scholes_price(s[i], k, remaining, r, q, sigma, option_type)
                    sample_records.append({"Step": i, "Spot": s[i], "Delta": new_delta, "Shares": shares, "Cash": cash, "Option": option_value})

        terminal_payoff = max(s[-1] - k, 0.0) if option_type.lower().startswith("c") else max(k - s[-1], 0.0)
        errors[j] = cash + shares * s[-1] - terminal_payoff
        total_costs[j] = path_cost

    return {
        "errors": errors,
        "costs": total_costs,
        "sample_path": pd.DataFrame(sample_records).set_index("Step"),
        "underlying_sample": s_paths[:, 0],
        "summary": pd.Series({
            "Initial option premium": premium,
            "Mean hedging error": float(np.mean(errors)),
            "Hedging error std": float(np.std(errors, ddof=1)),
            "Hedging error RMSE": math.sqrt(float(np.mean(errors**2))),
            "5% hedging error": float(np.quantile(errors, 0.05)),
            "95% hedging error": float(np.quantile(errors, 0.95)),
            "Mean transaction cost": float(np.mean(total_costs)),
            "Paths": paths,
            "Rebalances": rebalance_steps,
        }),
    }


# =============================================================================
# FINAL DIAGNOSTIC DASHBOARD
# =============================================================================


def pricing_diagnostics(
    s0: float,
    k: float,
    t: float,
    mu: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "Call",
    seed: int = 404,
) -> pd.DataFrame:
    mc = risk_neutral_monte_carlo(s0, k, t, r, q, sigma, option_type, paths=50_000, seed=seed)
    pq = p_vs_q_lab(s0, mu, r, q, sigma, t, paths=30_000, seed=seed + 1)
    hedge = delta_hedging_backtest(s0, k, t, mu, r, q, sigma, option_type, paths=500, rebalance_steps=max(12, int(63 * t)), seed=seed + 2)

    call = black_scholes_price(s0, k, t, r, q, sigma, "Call")
    put = black_scholes_price(s0, k, t, r, q, sigma, "Put")
    parity_left = call - put
    parity_right = s0 * math.exp(-q * t) - k * math.exp(-r * t)

    greeks = black_scholes_greeks(s0, k, t, r, q, sigma, option_type)
    bump = max(0.01, s0 * 1e-4)
    up = black_scholes_price(s0 + bump, k, t, r, q, sigma, option_type)
    down = black_scholes_price(s0 - bump, k, t, r, q, sigma, option_type)
    finite_delta = (up - down) / (2 * bump)

    rows = [
        ("Monte Carlo vs Black-Scholes", abs(mc["summary"]["MC - BS difference"]), "Closer to 0 is better"),
        ("Monte Carlo standard error", mc["summary"]["Monte Carlo standard error"], "Smaller is better"),
        ("Q martingale relative error", abs(pq["summary"]["Martingale relative error"]), "Closer to 0 is better"),
        ("Put-call parity residual", parity_left - parity_right, "Closer to 0 is better"),
        ("Analytical Delta", greeks["Delta"], "Black-Scholes sensitivity"),
        ("Finite-difference Delta", finite_delta, "Should match analytical Delta"),
        ("Delta discrepancy", finite_delta - greeks["Delta"], "Closer to 0 is better"),
        ("Mean hedge error", hedge["summary"]["Mean hedging error"], "Closer to 0 is better before costs/model error"),
        ("Hedge RMSE", hedge["summary"]["Hedging error RMSE"], "Smaller is better"),
    ]
    return pd.DataFrame(rows, columns=["Diagnostic", "Value", "Interpretation"]).set_index("Diagnostic")


# =============================================================================
# GUI HELPERS
# =============================================================================


def fmt(x, decimals=6):
    if isinstance(x, str):
        return x
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return str(x)


def run_self_tests() -> None:
    gateway = SmartAPIMarketData()
    assert gateway.normalize_symbol("NSE:INFY-EQ") == "INFY"
    assert gateway.normalize_symbol("SBIN.NS") == "SBIN"
    assert gateway._extract_search_rows({"data": [{"tradingsymbol": "INFY-EQ", "symboltoken": "1594"}]})[0]["symboltoken"] == "1594"
    history = make_demo_history(years=2, seed=1)
    stats = historical_statistics(history)
    assert stats["Annualized volatility (sigma)"] > 0

    bm = brownian_motion_lab(history)
    assert len(bm["W"]) > 100
    assert 0.5 < bm["diagnostics"]["QV ratio empirical/theoretical"] < 1.5

    t, paths = simulate_gbm_paths(100, 0.10, 0.20, 1.0, 252, 500, seed=2)
    assert paths.shape == (253, 500)
    assert np.all(paths > 0)

    ito = ito_lemma_lab(100, 0.10, 0.20, function_name="ln(S)", seed=3)
    assert np.isfinite(ito["summary"]["Ito approximation RMSE"])

    pq = p_vs_q_lab(100, 0.12, 0.06, 0.01, 0.20, years=1, paths=20_000, seed=4)
    assert abs(pq["summary"]["Martingale relative error"]) < 0.03

    bs = black_scholes_price(100, 100, 1, 0.05, 0.0, 0.20, "Call")
    assert abs(bs - 10.4506) < 0.01

    mc = risk_neutral_monte_carlo(100, 100, 1, 0.05, 0.0, 0.20, "Call", paths=80_000, seed=5)
    assert abs(mc["summary"]["MC - BS difference"]) < 0.35

    greeks = black_scholes_greeks(100, 100, 1, 0.05, 0.0, 0.20, "Call")
    assert 0 < greeks["Delta"] < 1
    assert greeks["Gamma"] > 0

    hedge = delta_hedging_backtest(100, 100, 1, 0.10, 0.05, 0.0, 0.20, "Call", paths=100, rebalance_steps=52, seed=6)
    assert len(hedge["errors"]) == 100

    diag = pricing_diagnostics(100, 100, 0.5, 0.10, 0.05, 0.0, 0.20, "Call", seed=7)
    assert "Put-call parity residual" in diag.index

    print("All HELM risk-neutral laboratory self-tests passed.")




def run_smartapi_smoke_test() -> None:
    """Interactive real-market test; credentials stay in the local process only."""
    print("HELM SmartAPI real-price smoke test")
    print("Credentials are read locally and are not written to disk by this program.\n")
    api_key = os.environ.get("HELM_SMARTAPI_API_KEY") or getpass.getpass("API key: ")
    client_code = os.environ.get("HELM_SMARTAPI_CLIENT_CODE") or input("Client code: ").strip()
    pin = os.environ.get("HELM_SMARTAPI_PIN") or getpass.getpass("PIN/MPIN: ")
    totp_secret = os.environ.get("HELM_SMARTAPI_TOTP_SECRET") or getpass.getpass("TOTP secret: ")
    symbol = (os.environ.get("HELM_SMARTAPI_SYMBOL") or input("NSE symbol [INFY]: ").strip() or "INFY").upper()

    end = datetime.now().replace(hour=15, minute=30, second=0, microsecond=0)
    start = (end - timedelta(days=730)).replace(hour=9, minute=15)
    gateway = SmartAPIMarketData()
    try:
        session = gateway.login(api_key, client_code, pin, totp_secret)
        print(f"Connected: {session.client_code}")
        history, trading_symbol, token, snapshot = gateway.fetch_equity_history(
            symbol, "NSE", start, end, progress=lambda m: print(m)
        )
        stats = historical_statistics(history)
        ltp = snapshot.get("ltp") if isinstance(snapshot, dict) else None
        spot = float(ltp) if ltp not in (None, "") else float(stats["Last close"])
        print(f"Resolved: {trading_symbol} | token {token}")
        print(f"Historical observations: {len(history)}")
        print(f"Latest historical close: ₹{float(stats['Last close']):,.2f}")
        print(f"LTP/current spot used: ₹{spot:,.2f}")
        print(f"Estimated physical drift mu: {float(stats['Annualized physical drift (mu)']):.2%}")
        print(f"Estimated annual volatility sigma: {float(stats['Annualized volatility (sigma)']):.2%}")

        sigma = float(stats["Annualized volatility (sigma)"])
        r, q, t = 0.06, 0.0, 1.0
        bs = black_scholes_price(spot, spot, t, r, q, sigma, "Call")
        mc = risk_neutral_monte_carlo(spot, spot, t, r, q, sigma, "Call", paths=50_000, seed=2026)
        print(f"ATM 1Y Black-Scholes call (r=6%, q=0): ₹{bs:,.2f}")
        print(f"ATM 1Y Monte Carlo call: ₹{float(mc['summary']['Monte Carlo price']):,.2f}")
        print(f"MC - BS difference: ₹{float(mc['summary']['MC - BS difference']):,.4f}")
        print("\nSmartAPI real-price smoke test completed successfully.")
    finally:
        gateway.logout()


# =============================================================================
# TKINTER DESKTOP APPLICATION
# =============================================================================


def launch_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    class ScrollableFrame(ttk.Frame):
        def __init__(self, master, width=430):
            super().__init__(master)
            self.canvas = tk.Canvas(self, highlightthickness=0, width=width)
            self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
            self.inner = ttk.Frame(self.canvas)
            self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")
            self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
            self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window, width=e.width))
            self.canvas.bind_all("<MouseWheel>", self._wheel)

        def _wheel(self, event):
            try:
                self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except Exception:
                pass

    class LabApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("HELM Risk-Neutral Stochastic Calculus & Derivatives Pricing Laboratory")
            self.geometry("1600x930")
            self.minsize(1200, 760)
            self.protocol("WM_DELETE_WINDOW", self.on_close)

            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
            style.configure("Sub.TLabel", font=("Segoe UI", 10))
            style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))

            self.gateway = SmartAPIMarketData()
            self.history = make_demo_history()
            self.data_label = tk.StringVar(value="Synthetic demo history")
            self.symbol_label = tk.StringVar(value="DEMO")

            # SmartAPI credentials are kept only in memory by this application.
            self.api_key_var = tk.StringVar()
            self.client_code_var = tk.StringVar()
            self.pin_var = tk.StringVar()
            self.totp_secret_var = tk.StringVar()
            self.smart_status_var = tk.StringVar(value="SmartAPI: not connected")
            self.market_symbol_var = tk.StringVar(value="INFY")
            self.market_exchange_var = tk.StringVar(value="NSE")
            self.token_override_var = tk.StringVar()
            self.market_start_var = tk.StringVar(value=(pd.Timestamp.today().normalize() - pd.DateOffset(years=5)).strftime("%Y-%m-%d"))
            self.market_end_var = tk.StringVar(value=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))
            self.latest_ltp_var = tk.StringVar(value="LTP: —")

            # Shared model inputs
            st = historical_statistics(self.history)
            self.spot_var = tk.StringVar(value=f"{st['Last close']:.2f}")
            self.strike_var = tk.StringVar(value=f"{st['Last close']:.2f}")
            self.mu_var = tk.StringVar(value=f"{st['Annualized physical drift (mu)']:.6f}")
            self.sigma_var = tk.StringVar(value=f"{st['Annualized volatility (sigma)']:.6f}")
            self.r_var = tk.StringVar(value="0.06")
            self.q_var = tk.StringVar(value="0.00")
            self.t_var = tk.StringVar(value="1.00")
            self.option_var = tk.StringVar(value="Call")

            header = ttk.Frame(self, padding=(14, 10))
            header.pack(fill="x")
            ttk.Label(header, text="HELM Risk-Neutral Stochastic Calculus & Derivatives Pricing Laboratory", style="Title.TLabel").pack(anchor="w")
            ttk.Label(
                header,
                text="Historical prices → returns → Brownian motion → GBM → Ito calculus → P vs Q → risk-neutral pricing → Greeks → delta hedging",
                style="Sub.TLabel",
            ).pack(anchor="w", pady=(2, 0))

            connect = ttk.LabelFrame(self, text="Angel One SmartAPI — real market data", padding=(10, 6))
            connect.pack(fill="x", padx=14, pady=(5, 3))
            ttk.Label(connect, text="API key").grid(row=0, column=0, sticky="w")
            ttk.Entry(connect, textvariable=self.api_key_var, width=20, show="•").grid(row=0, column=1, padx=(4, 10))
            ttk.Label(connect, text="Client code").grid(row=0, column=2, sticky="w")
            ttk.Entry(connect, textvariable=self.client_code_var, width=15).grid(row=0, column=3, padx=(4, 10))
            ttk.Label(connect, text="PIN/MPIN").grid(row=0, column=4, sticky="w")
            ttk.Entry(connect, textvariable=self.pin_var, width=12, show="•").grid(row=0, column=5, padx=(4, 10))
            ttk.Label(connect, text="TOTP secret").grid(row=0, column=6, sticky="w")
            ttk.Entry(connect, textvariable=self.totp_secret_var, width=22, show="•").grid(row=0, column=7, padx=(4, 10))
            ttk.Button(connect, text="Connect", command=self.login_smartapi).grid(row=0, column=8, padx=4)
            ttk.Label(connect, textvariable=self.smart_status_var).grid(row=0, column=9, sticky="w", padx=(8, 0))

            market = ttk.LabelFrame(self, text="Retrieve real equity prices", padding=(10, 6))
            market.pack(fill="x", padx=14, pady=(0, 3))
            ttk.Label(market, text="Symbol").grid(row=0, column=0, sticky="w")
            ttk.Entry(market, textvariable=self.market_symbol_var, width=12).grid(row=0, column=1, padx=(4, 10))
            ttk.Label(market, text="Exchange").grid(row=0, column=2, sticky="w")
            ttk.Combobox(market, textvariable=self.market_exchange_var, values=["NSE", "BSE"], state="readonly", width=7).grid(row=0, column=3, padx=(4, 10))
            ttk.Label(market, text="Token override").grid(row=0, column=4, sticky="w")
            ttk.Entry(market, textvariable=self.token_override_var, width=11).grid(row=0, column=5, padx=(4, 10))
            ttk.Label(market, text="Start").grid(row=0, column=6, sticky="w")
            ttk.Entry(market, textvariable=self.market_start_var, width=11).grid(row=0, column=7, padx=(4, 8))
            ttk.Label(market, text="End").grid(row=0, column=8, sticky="w")
            ttk.Entry(market, textvariable=self.market_end_var, width=11).grid(row=0, column=9, padx=(4, 8))
            ttk.Button(market, text="Fetch daily history + LTP", command=self.fetch_smartapi_history).grid(row=0, column=10, padx=4)
            ttk.Button(market, text="Refresh LTP", command=self.refresh_smartapi_ltp).grid(row=0, column=11, padx=4)
            ttk.Label(market, textvariable=self.latest_ltp_var).grid(row=0, column=12, padx=(8, 0), sticky="w")

            toolbar = ttk.Frame(self, padding=(14, 4, 14, 8))
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="Load historical CSV", command=self.load_csv).pack(side="left")
            ttk.Button(toolbar, text="Reset demo data", command=self.reset_demo).pack(side="left", padx=6)
            ttk.Label(toolbar, textvariable=self.data_label).pack(side="left", padx=12)

            self.nb = ttk.Notebook(self)
            self.nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))

            self.tabs = {}
            for name in [
                "1. Historical Data",
                "2. Brownian Motion",
                "3. GBM",
                "4. Ito's Lemma",
                "5. P vs Q",
                "6. Monte Carlo",
                "7. Black-Scholes & Greeks",
                "8. Delta Hedging",
                "9. Diagnostics",
            ]:
                frame = ttk.Frame(self.nb)
                self.nb.add(frame, text=name)
                self.tabs[name] = frame

            self._build_data_tab()
            self._build_brownian_tab()
            self._build_gbm_tab()
            self._build_ito_tab()
            self._build_pq_tab()
            self._build_mc_tab()
            self._build_bs_tab()
            self._build_hedge_tab()
            self._build_diag_tab()
            self.refresh_data_tab()

        # ------------------------ common GUI builders ------------------------
        def split_tab(self, tab_name):
            tab = self.tabs[tab_name]
            paned = ttk.Panedwindow(tab, orient="horizontal")
            paned.pack(fill="both", expand=True)
            left_holder = ttk.Frame(paned)
            right = ttk.Frame(paned)
            paned.add(left_holder, weight=0)
            paned.add(right, weight=1)
            left = ScrollableFrame(left_holder, width=450)
            left.pack(fill="both", expand=True)
            return left.inner, right

        def add_section(self, parent, title, text=None):
            ttk.Label(parent, text=title, style="Header.TLabel").pack(anchor="w", pady=(10, 4))
            if text:
                ttk.Label(parent, text=text, wraplength=410, justify="left").pack(anchor="w", pady=(0, 5))

        def add_entry(self, parent, label, variable, width=18):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=27).pack(side="left")
            ttk.Entry(row, textvariable=variable, width=width).pack(side="left")

        def add_combo(self, parent, label, variable, values):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=27).pack(side="left")
            ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=16).pack(side="left")

        def add_common_option_inputs(self, parent, include_mu=False):
            self.add_entry(parent, "Spot S0", self.spot_var)
            self.add_entry(parent, "Strike K", self.strike_var)
            self.add_entry(parent, "Maturity T (years)", self.t_var)
            if include_mu:
                self.add_entry(parent, "Physical drift mu", self.mu_var)
            self.add_entry(parent, "Risk-free rate r", self.r_var)
            self.add_entry(parent, "Dividend yield q", self.q_var)
            self.add_entry(parent, "Volatility sigma", self.sigma_var)
            self.add_combo(parent, "Option type", self.option_var, ["Call", "Put"])

        def make_figure(self, parent, figsize=(10, 7)):
            fig = Figure(figsize=figsize, dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            return fig, canvas

        def make_tree(self, parent, columns=("Metric", "Value"), height=14):
            tree = ttk.Treeview(parent, columns=columns, show="headings", height=height)
            for c in columns:
                tree.heading(c, text=c)
                tree.column(c, width=210 if c == columns[0] else 160, anchor="w")
            y = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
            x = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
            tree.pack(fill="both", expand=True, pady=4)
            x.pack(fill="x")
            return tree

        def fill_series(self, tree, series):
            tree.delete(*tree.get_children())
            for k, v in pd.Series(series).items():
                tree.insert("", "end", values=(str(k), fmt(v)))

        def fill_df(self, tree, df):
            tree.delete(*tree.get_children())
            df2 = df.copy()
            for idx, row in df2.iterrows():
                vals = [str(idx)] + [fmt(v) for v in row.values]
                tree.insert("", "end", values=vals)

        def params(self):
            return {
                "s0": float(self.spot_var.get()),
                "k": float(self.strike_var.get()),
                "t": float(self.t_var.get()),
                "mu": float(self.mu_var.get()),
                "r": float(self.r_var.get()),
                "q": float(self.q_var.get()),
                "sigma": float(self.sigma_var.get()),
                "option_type": self.option_var.get(),
            }

        def safe(self, func):
            try:
                func()
            except Exception as exc:
                messagebox.showerror("HELM Laboratory", str(exc))

        def sync_historical_params(self):
            st = historical_statistics(self.history)
            self.spot_var.set(f"{st['Last close']:.4f}")
            self.strike_var.set(f"{st['Last close']:.4f}")
            self.mu_var.set(f"{st['Annualized physical drift (mu)']:.8f}")
            self.sigma_var.set(f"{st['Annualized volatility (sigma)']:.8f}")

        def run_async(self, label: str, worker: Callable, done: Callable):
            self.smart_status_var.set(label)
            def task():
                try:
                    result = worker()
                except Exception as exc:
                    self.after(0, lambda: self._async_error(exc))
                else:
                    self.after(0, lambda: done(result))
            threading.Thread(target=task, daemon=True).start()

        def _async_error(self, exc: Exception):
            self.smart_status_var.set(f"SmartAPI error: {exc}")
            messagebox.showerror("SmartAPI", str(exc))

        @staticmethod
        def _parse_market_date(text_value: str, end_of_day: bool = False) -> datetime:
            value = datetime.strptime(text_value.strip(), "%Y-%m-%d")
            return value.replace(hour=15, minute=30) if end_of_day else value.replace(hour=9, minute=15)

        def login_smartapi(self):
            def worker():
                return self.gateway.login(
                    self.api_key_var.get(),
                    self.client_code_var.get(),
                    self.pin_var.get(),
                    self.totp_secret_var.get(),
                )
            def done(session):
                self.smart_status_var.set(f"SmartAPI connected: {session.client_code}")
            self.run_async("Connecting to SmartAPI…", worker, done)

        def fetch_smartapi_history(self):
            symbol = self.market_symbol_var.get().strip().upper()
            exchange = self.market_exchange_var.get().strip().upper()
            token_override = self.token_override_var.get().strip()
            try:
                start = self._parse_market_date(self.market_start_var.get(), False)
                end = self._parse_market_date(self.market_end_var.get(), True)
            except Exception as exc:
                messagebox.showerror("Dates", f"Use YYYY-MM-DD dates.\n{exc}")
                return

            def progress(message: str):
                self.after(0, lambda m=message: self.smart_status_var.set(m))

            def worker():
                return self.gateway.fetch_equity_history(symbol, exchange, start, end, token_override, progress)

            def done(result):
                history, trading_symbol, token, snapshot = result
                self.history = history
                self.symbol_label.set(trading_symbol)
                ltp = snapshot.get("ltp") if isinstance(snapshot, dict) else None
                try:
                    ltp_value = float(ltp)
                except Exception:
                    ltp_value = float("nan")
                self.data_label.set(
                    f"SmartAPI daily history: {trading_symbol} | token {token} | {len(history)} observations"
                )
                self.sync_historical_params()
                if math.isfinite(ltp_value) and ltp_value > 0:
                    self.spot_var.set(f"{ltp_value:.4f}")
                    self.strike_var.set(f"{ltp_value:.4f}")
                    self.latest_ltp_var.set(f"LTP: ₹{ltp_value:,.2f}")
                else:
                    self.latest_ltp_var.set("LTP: unavailable; using last close")
                self.refresh_data_tab()
                st = historical_statistics(self.history)
                self.smart_status_var.set(
                    f"Loaded {trading_symbol}: μ={st['Annualized physical drift (mu)']:.2%}, "
                    f"σ={st['Annualized volatility (sigma)']:.2%}"
                )
                self.nb.select(self.tabs["1. Historical Data"])
            self.run_async(f"Fetching {symbol} daily candles…", worker, done)

        def refresh_smartapi_ltp(self):
            symbol = self.market_symbol_var.get().strip().upper()
            exchange = self.market_exchange_var.get().strip().upper()
            token_override = self.token_override_var.get().strip()
            def worker():
                trading_symbol, token = self.gateway.resolve_symbol(exchange, symbol, token_override)
                snapshot = self.gateway.ltp_snapshot(exchange, trading_symbol, token)
                return trading_symbol, token, snapshot
            def done(result):
                trading_symbol, token, snapshot = result
                ltp = float(snapshot.get("ltp"))
                self.spot_var.set(f"{ltp:.4f}")
                self.latest_ltp_var.set(f"LTP: ₹{ltp:,.2f}")
                self.smart_status_var.set(f"LTP refreshed: {trading_symbol} | token {token}")
            self.run_async(f"Refreshing {symbol} LTP…", worker, done)

        def on_close(self):
            try:
                self.gateway.logout()
            finally:
                self.destroy()

        # ------------------------ data tab ------------------------
        def _build_data_tab(self):
            left, right = self.split_tab("1. Historical Data")
            self.add_section(left, "Target", "Retrieve or load historical stock prices, convert them into returns, and estimate the physical-measure drift μ and volatility σ used by later stochastic models. With SmartAPI, the latest LTP is used as the current spot S0 while μ and σ come from historical daily candles.")
            self.add_section(left, "Code logic", "1) Sort prices by date. 2) Compute arithmetic returns and log returns. 3) Estimate daily mean and volatility. 4) Annualize them. 5) Carry S0, μ and σ into the rest of the pipeline.")
            ttk.Button(left, text="Refresh historical analysis", command=lambda: self.safe(self.refresh_data_tab)).pack(anchor="w", pady=8)
            ttk.Label(left, textvariable=self.data_label, wraplength=410).pack(anchor="w", pady=4)
            tree_holder = ttk.Frame(left)
            tree_holder.pack(fill="both", expand=True)
            self.data_tree = self.make_tree(tree_holder)
            self.data_fig, self.data_canvas = self.make_figure(right)

        def refresh_data_tab(self):
            st = historical_statistics(self.history)
            self.fill_series(self.data_tree, st)
            self.data_fig.clear()
            ax1 = self.data_fig.add_subplot(211)
            ax2 = self.data_fig.add_subplot(212)
            ax1.plot(self.history.index, self.history["Close"])
            ax1.set_title(f"Historical Closing Price — {self.symbol_label.get()}")
            ax1.set_ylabel("Price")
            ax1.grid(alpha=0.2)
            ax2.plot(self.history.index, self.history["Log_Return"])
            ax2.axhline(0, linewidth=1)
            ax2.set_title("Daily Log Returns")
            ax2.set_ylabel("Log return")
            ax2.grid(alpha=0.2)
            self.data_fig.tight_layout()
            self.data_canvas.draw()

        def load_csv(self):
            path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if not path:
                return
            def action():
                self.history = load_historical_csv(path)
                self.symbol_label.set(Path(path).stem)
                self.data_label.set(f"Historical CSV: {Path(path).name}")
                self.sync_historical_params()
                self.refresh_data_tab()
                self.nb.select(self.tabs["1. Historical Data"])
            self.safe(action)

        def reset_demo(self):
            self.history = make_demo_history()
            self.symbol_label.set("DEMO")
            self.data_label.set("Synthetic demo history")
            self.latest_ltp_var.set("LTP: —")
            self.sync_historical_params()
            self.refresh_data_tab()

        # ------------------------ Brownian tab ------------------------
        def _build_brownian_tab(self):
            left, right = self.split_tab("2. Brownian Motion")
            self.add_section(left, "Target", "Turn standardized historical return shocks into an empirical Brownian-motion approximation and test the properties that motivate stochastic calculus.")
            self.add_section(left, "Jargon", "dW = a tiny random Brownian shock. W(t) = accumulated shocks. Independent increments means one Brownian shock should not strongly predict the next. Quadratic variation is the key property behind (dW)^2 ≈ dt.")
            self.add_section(left, "Code logic", "Center and standardize log returns → scale by √dt → cumulatively sum them → inspect increment autocorrelation, normality and quadratic variation.")
            ttk.Button(left, text="Run Brownian laboratory", command=lambda: self.safe(self.run_brownian)).pack(anchor="w", pady=8)
            holder = ttk.Frame(left); holder.pack(fill="both", expand=True)
            self.bm_tree = self.make_tree(holder)
            self.bm_fig, self.bm_canvas = self.make_figure(right)

        def run_brownian(self):
            out = brownian_motion_lab(self.history)
            self.fill_series(self.bm_tree, out["diagnostics"])
            self.bm_fig.clear()
            ax1 = self.bm_fig.add_subplot(221); ax2 = self.bm_fig.add_subplot(222)
            ax3 = self.bm_fig.add_subplot(223); ax4 = self.bm_fig.add_subplot(224)
            ax1.plot(out["W"].index, out["W"].values); ax1.set_title("Empirical Brownian Path Approximation"); ax1.grid(alpha=0.2)
            ax2.hist(out["dW"].values, bins=40, density=True, alpha=0.75); ax2.set_title("Brownian Increment Distribution")
            vals = out["dW"].values
            ax3.scatter(vals[:-1], vals[1:], alpha=0.35); ax3.set_title("Increment t vs Increment t+1"); ax3.axhline(0, lw=1); ax3.axvline(0, lw=1)
            cumulative_qv = np.cumsum(vals**2)
            theo = np.arange(1, len(vals)+1) / TRADING_DAYS
            ax4.plot(cumulative_qv, label="Empirical QV"); ax4.plot(theo, label="Theoretical t"); ax4.set_title("Quadratic Variation"); ax4.legend(); ax4.grid(alpha=0.2)
            self.bm_fig.tight_layout(); self.bm_canvas.draw()

        # ------------------------ GBM tab ------------------------
        def _build_gbm_tab(self):
            left, right = self.split_tab("3. GBM")
            self.add_section(left, "Target", "Move from raw Brownian shocks to a positive stock-price model: dS = drift·S·dt + σ·S·dW.")
            self.add_section(left, "Code logic", "Use exact geometric-Brownian-motion steps so every simulated stock price stays positive. Under P the drift is μ; under Q the drift becomes r−q.")
            self.gbm_measure = tk.StringVar(value="Physical P")
            self.gbm_paths = tk.StringVar(value="1000")
            self.gbm_steps = tk.StringVar(value="252")
            self.add_entry(left, "Spot S0", self.spot_var)
            self.add_entry(left, "Physical drift mu", self.mu_var)
            self.add_entry(left, "Risk-free rate r", self.r_var)
            self.add_entry(left, "Dividend yield q", self.q_var)
            self.add_entry(left, "Volatility sigma", self.sigma_var)
            self.add_entry(left, "Years", self.t_var)
            self.add_entry(left, "Simulation paths", self.gbm_paths)
            self.add_entry(left, "Time steps", self.gbm_steps)
            self.add_combo(left, "Measure", self.gbm_measure, ["Physical P", "Risk-neutral Q"])
            ttk.Button(left, text="Simulate GBM", command=lambda: self.safe(self.run_gbm)).pack(anchor="w", pady=8)
            holder = ttk.Frame(left); holder.pack(fill="both", expand=True)
            self.gbm_tree = self.make_tree(holder)
            self.gbm_fig, self.gbm_canvas = self.make_figure(right)

        def run_gbm(self):
            p = self.params(); measure = self.gbm_measure.get()
            drift = p["mu"] if measure.startswith("Physical") else p["r"] - p["q"]
            time, paths = simulate_gbm_paths(p["s0"], drift, p["sigma"], p["t"], int(self.gbm_steps.get()), int(self.gbm_paths.get()), seed=100, antithetic=True)
            self.fill_series(self.gbm_tree, gbm_summary(paths))
            self.gbm_fig.clear(); ax1 = self.gbm_fig.add_subplot(211); ax2 = self.gbm_fig.add_subplot(212)
            nshow = min(60, paths.shape[1])
            ax1.plot(time, paths[:, :nshow], alpha=0.25); ax1.set_title(f"GBM Paths under {measure}"); ax1.set_xlabel("Years"); ax1.set_ylabel("Price"); ax1.grid(alpha=0.2)
            ax2.hist(paths[-1], bins=50, alpha=0.75); ax2.set_title("Terminal Price Distribution"); ax2.set_xlabel("Terminal price")
            self.gbm_fig.tight_layout(); self.gbm_canvas.draw()

        # ------------------------ Ito tab ------------------------
        def _build_ito_tab(self):
            left, right = self.split_tab("4. Ito's Lemma")
            self.add_section(left, "Target", "Show why ordinary calculus is incomplete for a random GBM path and why Ito's second-order correction is needed.")
            self.add_section(left, "Code logic", "Simulate one GBM path → choose f(S) → compare actual df with f′(S)dS → add ½f″(S)σ²S²dt → compare errors again.")
            self.ito_fn = tk.StringVar(value="ln(S)")
            self.ito_steps = tk.StringVar(value="252")
            self.add_entry(left, "Spot S0", self.spot_var)
            self.add_entry(left, "Physical drift mu", self.mu_var)
            self.add_entry(left, "Volatility sigma", self.sigma_var)
            self.add_entry(left, "Years", self.t_var)
            self.add_entry(left, "Steps", self.ito_steps)
            self.add_combo(left, "Function f(S)", self.ito_fn, list(ITO_FUNCTIONS.keys()))
            ttk.Button(left, text="Run Ito laboratory", command=lambda: self.safe(self.run_ito)).pack(anchor="w", pady=8)
            holder = ttk.Frame(left); holder.pack(fill="both", expand=True)
            self.ito_tree = self.make_tree(holder)
            self.ito_fig, self.ito_canvas = self.make_figure(right)

        def run_ito(self):
            p = self.params()
            out = ito_lemma_lab(p["s0"], p["mu"], p["sigma"], p["t"], int(self.ito_steps.get()), self.ito_fn.get())
            self.fill_series(self.ito_tree, out["summary"])
            f = out["frame"]
            self.ito_fig.clear(); ax1 = self.ito_fig.add_subplot(221); ax2 = self.ito_fig.add_subplot(222); ax3 = self.ito_fig.add_subplot(223); ax4 = self.ito_fig.add_subplot(224)
            ax1.plot(out["price_path"]); ax1.set_title("Simulated GBM Price Path"); ax1.grid(alpha=0.2)
            ax2.plot(f["Actual dF"].values, label="Actual dF", alpha=0.8); ax2.plot(f["Ito approx"].values, label="Ito approx", alpha=0.7); ax2.set_title("Actual vs Ito Change"); ax2.legend(); ax2.grid(alpha=0.2)
            ax3.hist(f["Ordinary error"], bins=35, alpha=0.65, label="Ordinary error"); ax3.hist(f["Ito error"], bins=35, alpha=0.55, label="Ito error"); ax3.set_title("Approximation Error"); ax3.legend()
            ax4.plot(f["Ito correction"].values); ax4.set_title("Ito Second-Order Correction"); ax4.axhline(0, lw=1); ax4.grid(alpha=0.2)
            self.ito_fig.tight_layout(); self.ito_canvas.draw()

        # ------------------------ P vs Q tab ------------------------
        def _build_pq_tab(self):
            left, right = self.split_tab("5. P vs Q")
            self.add_section(left, "Target", "Separate forecasting/risk under the physical measure P from arbitrage-free pricing under the risk-neutral measure Q.")
            self.add_section(left, "Code logic", "Simulate the same stock twice. P uses historical drift μ. Q replaces μ with r−q. Then test whether the discounted Q stock price has an expected value close to S0 (martingale condition).")
            self.pq_paths = tk.StringVar(value="30000")
            self.add_entry(left, "Spot S0", self.spot_var)
            self.add_entry(left, "Physical drift mu", self.mu_var)
            self.add_entry(left, "Risk-free rate r", self.r_var)
            self.add_entry(left, "Dividend yield q", self.q_var)
            self.add_entry(left, "Volatility sigma", self.sigma_var)
            self.add_entry(left, "Years", self.t_var)
            self.add_entry(left, "Paths", self.pq_paths)
            ttk.Button(left, text="Compare P and Q", command=lambda: self.safe(self.run_pq)).pack(anchor="w", pady=8)
            holder = ttk.Frame(left); holder.pack(fill="both", expand=True)
            self.pq_tree = self.make_tree(holder)
            self.pq_fig, self.pq_canvas = self.make_figure(right)

        def run_pq(self):
            p = self.params(); out = p_vs_q_lab(p["s0"], p["mu"], p["r"], p["q"], p["sigma"], p["t"], int(self.pq_paths.get()))
            self.fill_series(self.pq_tree, out["summary"])
            self.pq_fig.clear(); ax1 = self.pq_fig.add_subplot(211); ax2 = self.pq_fig.add_subplot(212)
            ax1.hist(out["p_terminal"], bins=55, alpha=0.55, density=True, label="Physical P"); ax1.hist(out["q_terminal"], bins=55, alpha=0.55, density=True, label="Risk-neutral Q"); ax1.set_title("Terminal Distribution: P vs Q"); ax1.legend()
            discounted = math.exp(-(p["r"] - p["q"]) * p["t"]) * out["q_terminal"]
            running = np.cumsum(discounted) / np.arange(1, len(discounted)+1)
            ax2.plot(running); ax2.axhline(p["s0"], linestyle="--", label="S0"); ax2.set_title("Q Martingale Test: Mean Discounted Terminal Price"); ax2.legend(); ax2.grid(alpha=0.2)
            self.pq_fig.tight_layout(); self.pq_canvas.draw()

        # ------------------------ Monte Carlo tab ------------------------
        def _build_mc_tab(self):
            left, right = self.split_tab("6. Monte Carlo")
            self.add_section(left, "Target", "Price a European option by simulating the terminal stock price under Q, computing the payoff, and discounting the average payoff back to today.")
            self.add_section(left, "Code logic", "Generate Z~N(0,1) → simulate ST with drift r−q → payoff=max(ST−K,0) or max(K−ST,0) → discount by e^(−rT) → compare with Black-Scholes → estimate standard error and 95% confidence interval.")
            self.mc_paths = tk.StringVar(value="100000")
            self.add_common_option_inputs(left)
            self.add_entry(left, "Monte Carlo paths", self.mc_paths)
            ttk.Button(left, text="Run risk-neutral Monte Carlo", command=lambda: self.safe(self.run_mc)).pack(anchor="w", pady=8)
            holder = ttk.Frame(left); holder.pack(fill="both", expand=True)
            self.mc_tree = self.make_tree(holder)
            self.mc_fig, self.mc_canvas = self.make_figure(right)

        def run_mc(self):
            p = self.params(); out = risk_neutral_monte_carlo(p["s0"], p["k"], p["t"], p["r"], p["q"], p["sigma"], p["option_type"], int(self.mc_paths.get()))
            conv = monte_carlo_convergence(p["s0"], p["k"], p["t"], p["r"], p["q"], p["sigma"], p["option_type"])
            self.fill_series(self.mc_tree, out["summary"])
            self.mc_fig.clear(); ax1 = self.mc_fig.add_subplot(211); ax2 = self.mc_fig.add_subplot(212)
            ax1.hist(out["terminal"], bins=55, alpha=0.75); ax1.axvline(p["k"], linestyle="--", label="Strike"); ax1.set_title("Risk-Neutral Terminal Stock Distribution"); ax1.legend()
            ax2.plot(conv["Paths"], conv["MC Price"], marker="o", label="Monte Carlo"); ax2.axhline(conv["BS Price"].iloc[0], linestyle="--", label="Black-Scholes"); ax2.set_xscale("log"); ax2.set_title("Monte Carlo Convergence"); ax2.set_xlabel("Paths (log scale)"); ax2.set_ylabel("Option price"); ax2.legend(); ax2.grid(alpha=0.2)
            self.mc_fig.tight_layout(); self.mc_canvas.draw()

        # ------------------------ Black-Scholes/Greeks tab ------------------------
        def _build_bs_tab(self):
            left, right = self.split_tab("7. Black-Scholes & Greeks")
            self.add_section(left, "Target", "Benchmark the risk-neutral Monte Carlo price with the closed-form Black-Scholes model and measure how option value reacts to spot, volatility, time and rates.")
            self.add_section(left, "Code logic", "Calculate d1/d2 → option value → analytical Delta, Gamma, Vega, Theta and Rho. Then sweep spot and volatility to visualize price and sensitivity changes.")
            self.add_common_option_inputs(left)
            ttk.Button(left, text="Price option & calculate Greeks", command=lambda: self.safe(self.run_bs)).pack(anchor="w", pady=8)
            holder = ttk.Frame(left); holder.pack(fill="both", expand=True)
            self.bs_tree = self.make_tree(holder)
            self.bs_fig, self.bs_canvas = self.make_figure(right)

        def run_bs(self):
            p = self.params(); g = black_scholes_greeks(p["s0"], p["k"], p["t"], p["r"], p["q"], p["sigma"], p["option_type"])
            self.fill_series(self.bs_tree, g)
            spots = np.linspace(max(0.2*p["s0"], 0.01), 1.8*p["s0"], 120)
            prices = [black_scholes_price(s, p["k"], p["t"], p["r"], p["q"], p["sigma"], p["option_type"]) for s in spots]
            deltas = [_bs_delta(s, p["k"], p["t"], p["r"], p["q"], p["sigma"], p["option_type"]) for s in spots]
            vols = np.linspace(max(0.03, 0.35*p["sigma"]), 1.8*p["sigma"], 100)
            vol_prices = [black_scholes_price(p["s0"], p["k"], p["t"], p["r"], p["q"], v, p["option_type"]) for v in vols]
            self.bs_fig.clear(); ax1 = self.bs_fig.add_subplot(221); ax2 = self.bs_fig.add_subplot(222); ax3 = self.bs_fig.add_subplot(223); ax4 = self.bs_fig.add_subplot(224)
            ax1.plot(spots, prices); ax1.axvline(p["k"], linestyle="--"); ax1.set_title("Option Price vs Spot"); ax1.grid(alpha=0.2)
            ax2.plot(spots, deltas); ax2.axvline(p["k"], linestyle="--"); ax2.set_title("Delta vs Spot"); ax2.grid(alpha=0.2)
            ax3.plot(vols, vol_prices); ax3.set_title("Option Price vs Volatility"); ax3.set_xlabel("Sigma"); ax3.grid(alpha=0.2)
            greek_names = ["Delta", "Gamma", "Vega (per 1 vol point)", "Theta (per trading day)", "Rho (per 1 rate point)"]
            ax4.bar(greek_names, [g[x] for x in greek_names]); ax4.set_title("Greeks Snapshot"); ax4.tick_params(axis="x", rotation=25)
            self.bs_fig.tight_layout(); self.bs_canvas.draw()

        # ------------------------ Delta hedging tab ------------------------
        def _build_hedge_tab(self):
            left, right = self.split_tab("8. Delta Hedging")
            self.add_section(left, "Target", "Test whether continuously re-estimating Black-Scholes Delta can hedge the price risk of a short European option along realized stock paths.")
            self.add_section(left, "Code logic", "Sell option → buy Delta shares → finance the hedge through cash → simulate stock under P → accrue cash and dividends → recalculate Delta → rebalance → subtract transaction costs → compare final hedge portfolio with option payoff.")
            self.hedge_paths = tk.StringVar(value="1000")
            self.hedge_steps = tk.StringVar(value="63")
            self.hedge_cost = tk.StringVar(value="5")
            self.add_common_option_inputs(left, include_mu=True)
            self.add_entry(left, "Hedging paths", self.hedge_paths)
            self.add_entry(left, "Rebalance steps", self.hedge_steps)
            self.add_entry(left, "Transaction cost (bps)", self.hedge_cost)
            ttk.Button(left, text="Run delta-hedging backtest", command=lambda: self.safe(self.run_hedge)).pack(anchor="w", pady=8)
            holder = ttk.Frame(left); holder.pack(fill="both", expand=True)
            self.hedge_tree = self.make_tree(holder)
            self.hedge_fig, self.hedge_canvas = self.make_figure(right)

        def run_hedge(self):
            p = self.params(); out = delta_hedging_backtest(p["s0"], p["k"], p["t"], p["mu"], p["r"], p["q"], p["sigma"], p["option_type"], int(self.hedge_paths.get()), int(self.hedge_steps.get()), float(self.hedge_cost.get()))
            self.fill_series(self.hedge_tree, out["summary"])
            sample = out["sample_path"]
            self.hedge_fig.clear(); ax1 = self.hedge_fig.add_subplot(221); ax2 = self.hedge_fig.add_subplot(222); ax3 = self.hedge_fig.add_subplot(223); ax4 = self.hedge_fig.add_subplot(224)
            ax1.plot(out["underlying_sample"]); ax1.axhline(p["k"], linestyle="--", label="Strike"); ax1.set_title("Sample Realized Stock Path"); ax1.legend(); ax1.grid(alpha=0.2)
            ax2.plot(sample.index, sample["Delta"]); ax2.set_title("Dynamic Delta"); ax2.set_ylim(-1.05, 1.05); ax2.grid(alpha=0.2)
            ax3.hist(out["errors"], bins=45, alpha=0.75); ax3.axvline(0, linestyle="--"); ax3.set_title("Terminal Hedging Error Distribution")
            ax4.scatter(out["costs"], out["errors"], alpha=0.35); ax4.set_title("Transaction Costs vs Hedge Error"); ax4.set_xlabel("Cost"); ax4.set_ylabel("Hedge error")
            self.hedge_fig.tight_layout(); self.hedge_canvas.draw()

        # ------------------------ diagnostics tab ------------------------
        def _build_diag_tab(self):
            left, right = self.split_tab("9. Diagnostics")
            self.add_section(left, "Target", "Validate that the entire risk-neutral pricing pipeline is internally consistent rather than trusting one model output blindly.")
            self.add_section(left, "Code logic", "Check Monte Carlo against Black-Scholes, discounted-Q martingale behaviour, put-call parity, analytical versus finite-difference Delta, and the distribution of delta-hedging errors.")
            self.add_common_option_inputs(left, include_mu=True)
            ttk.Button(left, text="Run full diagnostics", command=lambda: self.safe(self.run_diag)).pack(anchor="w", pady=8)
            self.diag_status = tk.StringVar(value="Run the diagnostics to validate the pricing pipeline.")
            ttk.Label(left, textvariable=self.diag_status, wraplength=410).pack(anchor="w", pady=5)
            # custom tree 3 columns
            holder = ttk.Frame(left); holder.pack(fill="both", expand=True)
            self.diag_tree = ttk.Treeview(holder, columns=("Diagnostic", "Value", "Interpretation"), show="headings", height=16)
            for c, w in [("Diagnostic", 220), ("Value", 120), ("Interpretation", 280)]:
                self.diag_tree.heading(c, text=c); self.diag_tree.column(c, width=w, anchor="w")
            self.diag_tree.pack(fill="both", expand=True)
            self.diag_fig, self.diag_canvas = self.make_figure(right)

        def run_diag(self):
            p = self.params(); df = pricing_diagnostics(p["s0"], p["k"], p["t"], p["mu"], p["r"], p["q"], p["sigma"], p["option_type"])
            self.diag_tree.delete(*self.diag_tree.get_children())
            for idx, row in df.iterrows():
                self.diag_tree.insert("", "end", values=(idx, fmt(row["Value"]), row["Interpretation"]))

            mc_diff = float(df.loc["Monte Carlo vs Black-Scholes", "Value"])
            mart = float(df.loc["Q martingale relative error", "Value"])
            parity = abs(float(df.loc["Put-call parity residual", "Value"]))
            delta_gap = abs(float(df.loc["Delta discrepancy", "Value"]))
            self.diag_status.set(
                f"Core validation: |MC-BS|={mc_diff:.4f}, |martingale error|={mart:.4%}, "
                f"|parity residual|={parity:.6f}, |delta gap|={delta_gap:.6f}."
            )

            labels = ["MC-BS", "Martingale", "Parity", "Delta gap"]
            values = [mc_diff, mart, parity, delta_gap]
            self.diag_fig.clear(); ax1 = self.diag_fig.add_subplot(211); ax2 = self.diag_fig.add_subplot(212)
            ax1.bar(labels, values); ax1.set_title("Core Numerical Consistency Errors"); ax1.tick_params(axis="x", rotation=15)
            conv = monte_carlo_convergence(p["s0"], p["k"], p["t"], p["r"], p["q"], p["sigma"], p["option_type"])
            ax2.plot(conv["Paths"], conv["Abs Error"], marker="o"); ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_title("Monte Carlo Absolute Error vs Simulation Count"); ax2.set_xlabel("Paths"); ax2.set_ylabel("Absolute pricing error"); ax2.grid(alpha=0.2)
            self.diag_fig.tight_layout(); self.diag_canvas.draw()

    app = LabApp()
    app.mainloop()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        run_self_tests()
    elif "--smartapi-smoke-test" in sys.argv:
        run_smartapi_smoke_test()
    else:
        launch_gui()
