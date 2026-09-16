"""
tests/test_loader.py

Basic unit tests for src.data.loader -- covering the happy path (loading a valid CSV)
and the error paths (missing file, unknown extension) that `load_data()` is supposed to
handle explicitly rather than fail confusingly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from src.data.loader import load_data, data_summary


@pytest.fixture
def sample_csv(tmp_path):
    """Write a tiny CSV with a missing value and a duplicate row, for summary checks."""
    df = pd.DataFrame({
        "age": [25, 30, None, 40],
        "city": ["Mumbai", "Delhi", "Mumbai", "Delhi"],
        "target": [0, 1, 0, 1],
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add one duplicate row
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_load_data_reads_csv_correctly(sample_csv):
    df = load_data(str(sample_csv), show_summary=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (5, 3)
    assert list(df.columns) == ["age", "city", "target"]


def test_load_data_infers_type_from_extension(sample_csv):
    # no file_type passed -- should still work via extension inference
    df = load_data(str(sample_csv))
    assert len(df) == 5


def test_load_data_missing_file_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_data("this/path/does/not/exist.csv")


def test_load_data_unknown_extension_raises_valueerror(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("not really tabular data")
    with pytest.raises(ValueError):
        load_data(str(bad_file))


def test_data_summary_reports_missing_and_duplicates(sample_csv):
    df = load_data(str(sample_csv), show_summary=False)
    summary = data_summary(df)
    assert "age" in summary          # the column with a missing value should be listed
    assert "Duplicate rows: 1" in summary
