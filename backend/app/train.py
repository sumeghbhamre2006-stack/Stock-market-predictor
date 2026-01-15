import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

# ============================
# CONFIG
# ============================

STOCKS = [
    # Banks & Finance
    "HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
"INDUSINDBK.NS","BANKBARODA.NS","PNB.NS","FEDERALBNK.NS","IDFCFIRSTB.NS",
"CANBK.NS","UNIONBANK.NS","AUBANK.NS","RBLBANK.NS","YESBANK.NS",
"CHOLAFIN.NS","MUTHOOTFIN.NS","MANAPPURAM.NS","L&TFH.NS","HDFCAMC.NS",
"ICICIPRULI.NS","HDFCLIFE.NS","SBILIFE.NS","ICICIGI.NS","HDFCERGO.NS"
,

    # IT
    "TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","LTIM.NS",
"MPHASIS.NS","COFORGE.NS","PERSISTENT.NS","CYIENT.NS","ZENSARTECH.NS",
"SONATSOFTW.NS","KPITTECH.NS","TATAELXSI.NS","NEWGEN.NS","BSOFT.NS",
"HAPPSTMNDS.NS","AFFLE.NS","INTELLECT.NS","ROUTE.NS",

    # Energy & Infra
   "RELIANCE.NS","ONGC.NS","BPCL.NS","IOC.NS","HINDPETRO.NS","GAIL.NS",
"OIL.NS","NTPC.NS","POWERGRID.NS","TATAPOWER.NS","ADANIENT.NS",
"ADANIPORTS.NS","ADANIGREEN.NS","ADANIPOWER.NS","JSWENERGY.NS",
"NHPC.NS","SJVN.NS","TORNTPOWER.NS","IREDA.NS","SUZLON.NS",
"TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS","NMDC.NS",
"JINDALSTEL.NS","APLAPOLLO.NS","RATNAMANI.NS","HINDZINC.NS",
"NATIONALUM.NS","MOIL.NS","COALINDIA.NS","WELCORP.NS","JSL.NS"
"TATASTEEL.NS","JSWSTEEL.NS","HINDALCO.NS","VEDL.NS","SAIL.NS","NMDC.NS",
"JINDALSTEL.NS","APLAPOLLO.NS","RATNAMANI.NS","HINDZINC.NS",
"NATIONALUM.NS","MOIL.NS","COALINDIA.NS","WELCORP.NS","JSL.NS"
"HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS",
"MARICO.NS","COLPAL.NS","GODREJCP.NS","TATACONSUM.NS","UBL.NS",
"VBL.NS","RADICO.NS","EMAMILTD.NS","PGHH.NS","JUBLFOOD.NS",
"ABFRL.NS","TRENT.NS","DMART.NS","AVENUE.NS"
"MARUTI.NS","M&M.NS","BAJAJ-AUTO.NS","EICHERMOT.NS","TVSMOTOR.NS",
"HEROMOTOCO.NS","ASHOKLEY.NS","ESCORTS.NS","SONACOMS.NS","MOTHERSON.NS",
"BALKRISIND.NS","MRF.NS","CEATLTD.NS","AMARAJABAT.NS","EXIDEIND.NS",
"ENDURANCE.NS","BOSCHLTD.NS","TATAMTRDVR.NS"
"SUNPHARMA.NS","CIPLA.NS","DIVISLAB.NS","DRREDDY.NS","LUPIN.NS",
"AUROPHARMA.NS","ALKEM.NS","TORNTPHARM.NS","GLENMARK.NS","BIOCON.NS",
"APOLLOHOSP.NS","FORTIS.NS","MAXHEALTH.NS","LAURUSLABS.NS",
"METROPOLIS.NS","SYNGENE.NS","STAR.NS"
"LT.NS","SIEMENS.NS","ABB.NS","BHEL.NS","CUMMINSIND.NS","THERMAX.NS",
"HAVELLS.NS","VOLTAS.NS","KEI.NS","POLYCAB.NS","CGPOWER.NS",
"HAL.NS","BEL.NS","BEML.NS","GRINDWELL.NS","SKFINDIA.NS",
"APLAPOLLO.NS","IRCON.NS","RVNL.NS"
"DLF.NS","OBEROIRLTY.NS","PRESTIGE.NS","PHOENIXLTD.NS","BRIGADE.NS",
"SOBHA.NS","IRCTC.NS","GMRINFRA.NS","JSWINFRA.NS","CONCOR.NS",
"ADANITRANS.NS","AWL.NS","ZOMATO.NS","PAYTM.NS","NYKAA.NS",
"POLICYBZR.NS","DELHIVERY.NS","INDIGO.NS","IRFC.NS","NHPC.NS"


   
]


FEATURES = [
    "rsi",
    "macd",
    "roc",
    "ema_50",
    "trend_50",
    "adx",
    "atr",
    "volume_change"
]

MODEL_PATH = "trained_model.pkl"
SCALER_PATH = "scalers.pkl"

# ============================
# FETCH SINGLE STOCK
# ============================

def fetch_single_stock(symbol):
    df = yf.download(
        symbol,
        period="10y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty or len(df) < 300:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df["symbol"] = symbol
    return df

# ============================
# INDICATORS
# ============================

def add_indicators_single(df):
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    df["rsi"] = ta.momentum.RSIIndicator(close).rsi()
    df["macd"] = ta.trend.MACD(close).macd()
    df["roc"] = ta.momentum.ROCIndicator(close).roc()

    df["ema_50"] = ta.trend.EMAIndicator(close, 50).ema_indicator()
    df["trend_50"] = (df["Close"] > df["Close"].rolling(50).mean()).astype(int)

    df["adx"] = ta.trend.ADXIndicator(
        high=df["High"],
        low=df["Low"],
        close=close
    ).adx()

    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=close
    ).average_true_range()

    df["volume_change"] = volume.pct_change()

    df.dropna(inplace=True)
    return df

# ============================
# BUILD DATASET (THRESHOLDED TARGET)
# ============================

def build_dataset(stocks):
    all_data = []

    horizon = 3
    move_threshold = 0.01  # 🔥 1% strong move

    for symbol in stocks:
        print(f"Processing {symbol}...")

        df = fetch_single_stock(symbol)
        if df is None:
            print(f"⚠️ Skipping {symbol} (bad data)")
            continue

        df = add_indicators_single(df)
        if df.empty:
            continue

        # ------------------------------------
        # 🔥 STRONG MOVE TARGET (KEY CHANGE)
        # ------------------------------------
        df["future_close"] = df["Close"].shift(-horizon)
        df["future_return"] = (df["future_close"] - df["Close"]) / df["Close"]
        df["abs_return"] = df["future_return"].abs()

        # 1 = strong move, 0 = no strong move
        df["target"] = (df["abs_return"] >= move_threshold).astype(int)

        df.dropna(inplace=True)

        if len(df) < 300:
            continue

        print(f"✅ {symbol} usable rows (strong-move): {len(df)}")
        all_data.append(df)

    if not all_data:
        raise RuntimeError("❌ No valid stock data collected")

    return pd.concat(all_data)


# ============================
# CLEAN FEATURES
# ============================

def clean_features(df, features):
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df[features] = df[features].clip(lower=-10, upper=10)
    df.dropna(subset=features, inplace=True)
    return df

# ============================
# NORMALIZATION (PER STOCK)
# ============================

def normalize_features(df, features):
    scalers = {}
    normalized = []

    for symbol, group in df.groupby("symbol"):
        scaler = StandardScaler()
        group[features] = scaler.fit_transform(group[features])
        scalers[symbol] = scaler
        normalized.append(group)

    return pd.concat(normalized), scalers

# ============================
# SPLIT DATA
# ============================

def split_data(df, features):
    df = df.sort_index()

    X = df[features]
    y = df["target"]

    split = int(len(df) * 0.8)

    return (
        X.iloc[:split],
        X.iloc[split:],
        y.iloc[:split],
        y.iloc[split:]
    )

# ============================
# TRAIN BASE MODEL
# ============================

def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model

# ============================
# CONFIDENCE BUCKET ANALYSIS
# ============================

def confidence_bucket_analysis(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    df_results = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": preds,
        "confidence": confidences
    })

    buckets = [
        (0.50, 0.55),
        (0.55, 0.60),
        (0.60, 0.65),
        (0.65, 1.00)
    ]

    print("\n📊 CONFIDENCE BUCKET ANALYSIS")
    print("-" * 60)

    for low, high in buckets:
        bucket = df_results[
            (df_results["confidence"] >= low) &
            (df_results["confidence"] < high)
        ]

        if len(bucket) == 0:
            continue

        acc = (bucket["y_true"] == bucket["y_pred"]).mean()

        print(
            f"Confidence {int(low*100)}–{int(high*100)}% | "
            f"Samples: {len(bucket):6d} | "
            f"Accuracy: {acc:.3f}"
        )

# ============================
# MAIN
# ============================

def main():
    print("\n🔹 Building dataset...")
    df = build_dataset(STOCKS)

    print("\n🔹 Cleaning features...")
    df = clean_features(df, FEATURES)

    print("\n🔹 Normalizing features...")
    df, scalers = normalize_features(df, FEATURES)

    print("\n🔹 Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, FEATURES)

    print("\n🔹 Training base model...")
    base_model = train_model(X_train, y_train)

    print("\n🔹 Calibrating probabilities...")
    model = CalibratedClassifierCV(
    base_model,
    method="isotonic",
    cv=3
)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n📊 Overall Accuracy:", accuracy_score(y_test, preds))
    print("\n📄 Classification Report:\n")
    print(classification_report(y_test, preds,zero_division=0))

    confidence_bucket_analysis(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scalers, SCALER_PATH)

    print("\n✅ Model saved as:", MODEL_PATH)
    print("✅ Scalers saved as:", SCALER_PATH)

if __name__ == "__main__":
    main()
