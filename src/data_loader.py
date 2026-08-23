from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_PATH = Path("data/credit_default.csv")


def download_credit_default() -> pd.DataFrame:
    """Download UCI Default of Credit Card Clients (dataset id 350).

    The target is normalized to a binary column named `default` so the rest of
    the project can use a consistent interface.
    """
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if y.shape[1] != 1:
        raise ValueError("Expected a single target column from UCI dataset 350")

    target = y.iloc[:, 0].astype(int).rename("default")
    df = pd.concat([X.reset_index(drop=True), target.reset_index(drop=True)], axis=1)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def load_or_download(path: Path = DATA_PATH) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return download_credit_default()


if __name__ == "__main__":
    df = load_or_download()
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
    print(df["default"].value_counts(normalize=True).rename("share"))
