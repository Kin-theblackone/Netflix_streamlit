"""Các kiểm thử nhỏ trên dữ liệu tự dựng (không giả mạo số liệu Kaggle)."""
from io import BytesIO
import sys
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from netflix_core import (
    build_tables, clean_netflix, fit_kmeans_pca, format_itemset,
    matching_titles, mine_apriori, read_netflix_csv, split_values,
)


def sample_df(n=24):
    rows = []
    for i in range(n):
        rows.append({
            "show_id": f"s{i+1}",
            "type": "Movie" if i % 3 else "TV Show",
            "title": f"Demo title {i}",
            "director": None,
            "cast": None,
            "country": "United States, Canada" if i % 2 else "India",
            "date_added": "September 25, 2021",
            "release_year": 2018 + i % 4,
            "rating": "TV-MA" if i % 2 else "PG",
            "duration": "1 Season" if i % 3 == 0 else "100 min",
            "listed_in": "Dramas, International Movies" if i % 2 else "Comedies, Dramas",
            "description": "Dòng kiểm thử",
        })
    return pd.DataFrame(rows)


def test_split_values():
    assert split_values("Dramas, Dramas, Comedy, comedy , ") == ["Dramas", "Comedy"]
    assert split_values(np.nan) == []


def test_read_raw_csv_and_zip():
    content = sample_df(6).to_csv(index=False).encode("utf-8")
    raw = read_netflix_csv(content, "netflix_titles.csv")
    assert len(raw) == 6
    bio = BytesIO()
    with zipfile.ZipFile(bio, mode="w") as zipf:
        zipf.writestr("inside/netflix_titles.csv", content)
    zipped = read_netflix_csv(bio.getvalue(), "netflix-shows.zip")
    assert zipped.equals(raw)


def test_clean_and_transactions():
    raw = sample_df(12)
    raw.loc[0, "listed_in"] = " Dramas, Comedies, dramas, "
    raw = pd.concat([raw, raw.iloc[[1]]], ignore_index=True)
    clean = clean_netflix(raw)
    assert len(clean) == 12
    assert clean.loc[0, "Genre"] == ["Dramas", "Comedies"]
    assert clean.loc[0, "type"] == "TV SHOW"
    assert pd.api.types.is_datetime64_any_dtype(clean["date_added"])
    genre, country, transactions, basket = build_tables(clean)
    assert len(genre) >= len(clean)
    assert len(country) >= len(clean)
    assert "Genre=Dramas" in basket.columns
    assert "Type=MOVIE" in basket.columns
    assert basket.loc["s1", "Type=TV SHOW"]
    assert "Country=India" in transactions.iloc[0]["transaction_items"]


def test_missing_columns_error():
    with pytest.raises(ValueError, match="thiếu cột"):
        clean_netflix(pd.DataFrame({"show_id": ["s1"]}))


def test_kmeans_pca_jaccard():
    clean = clean_netflix(sample_df(24))
    _, _, _, basket = build_tables(clean)
    labels, points, ratio = fit_kmeans_pca(basket, 3)
    assert len(labels) == 24
    assert points.shape == (24, 2)
    assert len(ratio) == 2
    recs = matching_titles(basket, "s1", limit=5)
    assert not recs.empty
    assert "s1" not in set(recs["show_id"])
    assert recs["similarity"].between(0, 1).all()


def test_apriori_if_mlxtend_present():
    pytest.importorskip("mlxtend")
    clean = clean_netflix(sample_df(24))
    _, _, _, basket = build_tables(clean)
    frequent, rules = mine_apriori(basket, .20, .50)
    assert not frequent.empty
    assert not rules.empty
    assert rules["confidence"].ge(.5).all()
    assert rules["support"].ge(.2).all()
    assert isinstance(format_itemset(frequent.iloc[0]["itemsets"]), str)
