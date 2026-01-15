from xgboost import XGBClassifier
import joblib

def train_model(df):
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    features = ['rsi', 'macd', 'sma_20']
    X = df[features]
    y = df['target']

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1
    )
    model.fit(X, y)

    joblib.dump(model, "trained_model.pkl")
    return model
