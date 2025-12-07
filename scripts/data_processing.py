# -----------------------------------------------
# data_processing.py
# -----------------------------------------------

from pathlib import Path
from typing import Iterable, List

# Importing necessary libraries
import pandas as pd              # Pandas: used for loading data and cleaning
import numpy as np               # Numpy: used for numerical operations
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ------------------------------------------------
# FUNCTION 1: Load and Clean Data
# ------------------------------------------------
def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file with minimal validation and duplicate removal."""

    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV is empty: {filepath}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read CSV: {filepath}") from exc

    if data.empty:
        raise ValueError(f"Dataset loaded but is empty: {filepath}")

    data = data.drop_duplicates(keep="first")
    return data  # returns cleaned dataset


# ------------------------------------------------
# FUNCTION 2: Encoding Categorical Variables
# ------------------------------------------------
def _validate_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")
    return list(columns)


def encoder(method: str, dataframe: pd.DataFrame, columns_label, columns_onehot):

    # ---------------------------
    # METHOD 1: LABEL ENCODER
    # ---------------------------
    if method == 'labelEncoder':
        df_lbl = dataframe.copy()
        cols = _validate_columns(df_lbl, columns_label)

        for col in cols:
            try:
                label = LabelEncoder()
                label.fit(list(df_lbl[col].values))
                df_lbl[col] = label.transform(df_lbl[col].values)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed label encoding for column '{col}'") from exc

        return df_lbl
    
    # ---------------------------
    # METHOD 2: ONE-HOT ENCODER
    # ---------------------------
    elif method == 'oneHotEncoder':
        df_oh = dataframe.copy()
        cols = _validate_columns(df_oh, columns_onehot)

        try:
            df_oh = pd.get_dummies(
                data=df_oh,
                prefix='ohe',
                prefix_sep='_',
                columns=cols,
                drop_first=True,
                dtype='int8'
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed one-hot encoding") from exc

        return df_oh


# ------------------------------------------------
# FUNCTION 3: SCALING NUMERICAL VARIABLES
# ------------------------------------------------
def scaler(method: str, data: pd.DataFrame, columns_scaler):

    # ---------------------------
    # METHOD 1: STANDARD SCALER
    # ---------------------------
    if method == 'standardScaler':
        df_standard = data.copy()
        cols = _validate_columns(df_standard, columns_scaler)
        try:
            df_standard[cols] = StandardScaler().fit_transform(df_standard[cols])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed standard scaling") from exc

        return df_standard
        
    # ---------------------------
    # METHOD 2: MIN-MAX SCALER
    # ---------------------------
    elif method == 'minMaxScaler':
        df_minmax = data.copy()
        cols = _validate_columns(df_minmax, columns_scaler)
        try:
            df_minmax[cols] = MinMaxScaler().fit_transform(df_minmax[cols])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed min-max scaling") from exc

        return df_minmax
    
    # ---------------------------
    # METHOD 3: LOG TRANSFORMATION
    # ---------------------------
    elif method == 'npLog':
        df_nplog = data.copy()
        cols = _validate_columns(df_nplog, columns_scaler)
        if (df_nplog[cols] <= 0).any().any():
            raise ValueError("Log transform requires strictly positive values")
        try:
            df_nplog[cols] = np.log(df_nplog[cols])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed log transform") from exc
        return df_nplog

    raise ValueError(f"Unsupported scaling method: {method}")
