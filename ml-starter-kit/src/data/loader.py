"""
src/data/loader.py

WHY?
    Every ML project starts the same way: read a file into a DataFrame, then spend the
    first ten minutes eyeballing `.shape`, `.info()`, `.isna().sum()` and `.duplicated().sum()`
    by hand. Doing that by hand, differently, in every notebook is exactly the kind of
    repetition this framework exists to remove.

WHAT?
    `load_data()` reads a CSV, Excel, or Parquet file into a pandas DataFrame, inferring the
    file type from its extension when you don't specify one, and prints a short, consistent
    data-quality summary (shape, dtypes, missing values, duplicates) every time.

HOW?
    Pass a path (and optionally a file_type override). The function validates the path exists,
    dispatches to the right pandas reader, and returns the DataFrame -- exactly the object you'd
    get from calling `pd.read_csv()` yourself, just with the boilerplate and the sanity checks
    already done for you.
"""
from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_TYPES = {"csv", "excel", "parquet"}
_EXTENSION_MAP = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def _infer_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in _EXTENSION_MAP:
        raise ValueError(
            f"Could not infer file type from extension '{suffix}'. "
            f"Pass file_type explicitly: one of {sorted(_SUPPORTED_TYPES)}."
        )
    return _EXTENSION_MAP[suffix]


def load_data(path: str, file_type: str | None = None, show_summary: bool = True, **read_kwargs) -> pd.DataFrame:
    """Load a dataset from CSV, Excel, or Parquet into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the data file.
    file_type : {"csv", "excel", "parquet"}, optional
        Force a specific reader instead of inferring it from the file extension.
    show_summary : bool, default True
        If True, print a short data-quality summary after loading.
    **read_kwargs :
        Extra keyword arguments forwarded to the underlying pandas reader
        (e.g. `sep=";"` for `read_csv`, `sheet_name=0` for `read_excel`).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If the file type can't be inferred and wasn't provided, or is unsupported.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No file found at '{path}'. Check the path and try again.")

    resolved_type = file_type or _infer_file_type(file_path)
    if resolved_type not in _SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file_type '{resolved_type}'. Choose one of {sorted(_SUPPORTED_TYPES)}.")

    if resolved_type == "csv":
        df = pd.read_csv(file_path, **read_kwargs)
    elif resolved_type == "excel":
        df = pd.read_excel(file_path, **read_kwargs)
    else:  # parquet
        df = pd.read_parquet(file_path, **read_kwargs)

    logger.info(f"Loaded '{file_path.name}' as {resolved_type} -> shape={df.shape}")

    if show_summary:
        print(data_summary(df))

    return df


def data_summary(df: pd.DataFrame) -> str:
    """Build a short, human-readable data-quality summary string for a DataFrame.

    Includes row/column counts, dtype breakdown, per-column missing-value counts
    (only for columns that actually have missing values), and the duplicate-row count.
    """
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    n_duplicates = df.duplicated().sum()

    lines = [
        "=" * 50,
        "DATA SUMMARY",
        "=" * 50,
        f"Rows: {n_rows:,}   Columns: {n_cols}",
        f"Duplicate rows: {n_duplicates:,}",
        "",
        "Dtypes:",
        df.dtypes.value_counts().to_string(),
        "",
    ]
    if len(missing) > 0:
        lines.append("Missing values (columns with at least one):")
        lines.append(missing.to_string())
    else:
        lines.append("Missing values: none")
    lines.append("=" * 50)
    return "\n".join(lines)
