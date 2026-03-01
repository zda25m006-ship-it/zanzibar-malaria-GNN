"""
Individual-level travel risk scorer.

Trains a Logistic Regression on per-patient pre-diagnosis attributes to
predict P(imported | attributes). Key design constraints:
  - Fitted ONLY on training-fold patients to prevent leakage
  - Uses only attributes known before diagnosis (not travel destination)
  - Scores are used as features for the GNN, not as the final prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score


# Columns used as features for risk scoring
# — all known before the positive test result
RISK_FEATURE_COLS = [
    'agegroup',
    'sex',
    'occupation',
    'home_district',
    'travel_outside_shehia',     # traveled within Zanzibar?
    'travel_over_4_nights',      # duration proxy (available for all tested)
    'travel_over_14_nights',     # long-stay proxy
    'months_on_zb',              # mainland origin (longer stay = more likely mainlander)
    'mainlander_on_zb',          # direct flag
]


class IndividualRiskScorer:
    """
    Per-fold Logistic Regression producing individual importation risk scores.

    Usage:
        scorer = IndividualRiskScorer()
        scorer.fit(train_patients_df)         # only training patients
        scores = scorer.predict_risk(test_patients_df)  # [0,1] per patient
        auc = scorer.auc_on(test_patients_df)           # self-check
    """

    def __init__(self, C: float = 0.1, max_iter: int = 500):
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()
        self.encoders: dict = {}
        self.feature_names: list = []
        self.fitted = False

    # ------------------------------------------------------------------
    def _build_X(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Build feature matrix from patient dataframe."""
        parts = []

        cat_cols = ['agegroup', 'sex', 'occupation', 'home_district', 'months_on_zb']
        num_cols = ['travel_outside_shehia', 'travel_over_4_nights',
                    'travel_over_14_nights', 'mainlander_on_zb']

        # Categorical → one-hot via label encoding then dummy expansion
        for col in cat_cols:
            col_data = df[col].fillna('Unknown').astype(str)
            if fit:
                enc = LabelEncoder()
                enc.fit(col_data)
                self.encoders[col] = enc
            enc = self.encoders.get(col)
            if enc is None:
                continue
            # Map unseen categories to 'Unknown' if possible
            known = set(enc.classes_)
            col_data = col_data.apply(lambda x: x if x in known else 'Unknown')
            idx = enc.transform(col_data)
            # One-hot
            n_classes = len(enc.classes_)
            ohe = np.zeros((len(df), n_classes))
            ohe[np.arange(len(df)), idx] = 1.0
            parts.append(ohe)

        # Numeric → fill missing with training median
        for col in num_cols:
            vals = df[col].copy()
            if fit:
                self._medians = getattr(self, '_medians', {})
                self._medians[col] = vals.median()
            med = getattr(self, '_medians', {}).get(col, 0.0)
            parts.append(vals.fillna(med).values.reshape(-1, 1))

        X = np.hstack(parts).astype(np.float32)

        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame) -> 'IndividualRiskScorer':
        """Fit scorer on training patients. Target = travel (1=imported)."""
        df = train_df.dropna(subset=['travel'])
        X = self._build_X(df, fit=True)
        y = df['travel'].values.astype(int)

        if y.sum() < 5 or (y == 0).sum() < 5:
            # Not enough signal — fall back to naive score
            self.model = None
            self._base_rate = y.mean()
            self.fitted = True
            return self

        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            class_weight='balanced', solver='lbfgs'
        )
        self.model.fit(X, y)
        self.fitted = True
        return self

    # ------------------------------------------------------------------
    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(imported) for each patient row."""
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        if self.model is None:
            return np.full(len(df), self._base_rate)
        X = self._build_X(df, fit=False)
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    def auc_on(self, df: pd.DataFrame) -> float:
        """Compute AUC on a held-out patient set (for diagnostics)."""
        df = df.dropna(subset=['travel'])
        y = df['travel'].values.astype(int)
        if len(np.unique(y)) < 2:
            return float('nan')
        scores = self.predict_risk(df)
        return roc_auc_score(y, scores)


# ------------------------------------------------------------------
def aggregate_risk_features(
    patient_df: pd.DataFrame,
    scorer: IndividualRiskScorer,
    year_month: str,
    districts: list,
    long_stay_shift: bool = True,
) -> dict:
    """
    Aggregate individual risk scores into per-district features for a month.

    Timing correction:
        Patients with travel_over_14_nights=1 likely traveled to mainland
        in the PREVIOUS month — optionally shift those to t-1 instead.
        This function handles current month only; the caller accumulates
        both months to build lag features.

    Returns:
        dict: {district -> {'mean_risk', 'frac_high_risk', 'n_total', 'n_high_risk',
                            'mean_risk_longstay', 'n_longstay'}}
    """
    month_patients = patient_df[patient_df['date'] == year_month].copy()
    scores = scorer.predict_risk(month_patients) if len(month_patients) > 0 else np.array([])
    month_patients = month_patients.copy()
    month_patients['risk_score'] = scores if len(scores) > 0 else 0.0

    result = {}
    for dist in districts:
        d_pat = month_patients[month_patients['home_district'] == dist]

        if long_stay_shift:
            # Short-stay patients: attributed to current month
            short = d_pat[d_pat['travel_over_14_nights'].fillna(0) == 0]
        else:
            short = d_pat

        n_total = len(d_pat)
        n_short = len(short)
        n_long  = n_total - n_short

        if len(short) > 0:
            mean_risk = short['risk_score'].mean()
            frac_high = (short['risk_score'] > 0.5).mean()
            n_high    = (short['risk_score'] > 0.5).sum()
        else:
            mean_risk = 0.0
            frac_high = 0.0
            n_high    = 0

        result[dist] = {
            'mean_risk':      float(mean_risk),
            'frac_high_risk': float(frac_high),
            'n_total':        float(n_total),
            'n_high_risk':    float(n_high),
            'n_longstay':     float(n_long),
        }
    return result
