"""Xử lý dữ liệu Netflix: giữ quy trình từ notebook đồ_án_netflix.ipynb.

Dataset: https://www.kaggle.com/datasets/shivamb/netflix-shows
Mỗi hàng là MỘT tựa phim/chương trình, không phải lịch sử xem của người dùng.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

REQUIRED_COLUMNS = {
    "show_id", "type", "title", "country", "date_added",
    "release_year", "rating", "listed_in",
}


def read_netflix_csv(contents: bytes, filename: str) -> pd.DataFrame:
    """Đọc CSV hoặc ZIP chứa CSV Kaggle, không phụ thuộc đường dẫn Colab."""
    filename = filename.lower().strip()
    if filename.endswith(".zip"):
        with zipfile.ZipFile(BytesIO(contents)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
            if not csv_names:
                raise ValueError("File ZIP không chứa dữ liệu .csv")
            preferred = next((name for name in csv_names if Path(name).name.lower() == "netflix_titles.csv"), csv_names[0])
            with zf.open(preferred) as file:
                return pd.read_csv(file, dtype={"show_id": "string"})
    if filename.endswith(".csv"):
        return pd.read_csv(BytesIO(contents), dtype={"show_id": "string"})
    raise ValueError("Hãy chọn file .csv hoặc .zip chứa netflix_titles.csv")


def split_values(value: object) -> list[str]:
    """Tách item theo dấu phẩy, bỏ rỗng và loại trùng không phân biệt chữ hoa."""
    if value is None or not isinstance(value, (str, list)) and pd.isna(value):
        return []
    chunks = value if isinstance(value, list) else str(value).split(",")
    cleaned, seen = [], set()
    for chunk in chunks:
        item = str(chunk).strip()
        marker = item.casefold()
        if item and marker not in seen:
            cleaned.append(item)
            seen.add(marker)
    return cleaned


def clean_netflix(raw: pd.DataFrame) -> pd.DataFrame:
    """Giữ đầy đủ bước chuẩn hóa/transaction giống notebook."""
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()
    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_cols:
        raise ValueError("Dữ liệu thiếu cột bắt buộc: " + ", ".join(missing_cols))

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["show_id"], inplace=True)
    df.drop_duplicates(subset=["show_id"], keep="first", inplace=True)
    for col in ("type", "rating"):
        df[col] = df[col].map(lambda x: x.upper() if isinstance(x, str) else x)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce", format="mixed")
    df["Genre"] = df["listed_in"].apply(split_values)
    df["Country"] = df["country"].apply(split_values)

    def items(row: pd.Series) -> list[str]:
        transaction = (
            [f"Genre={v}" for v in row["Genre"]]
            + [f"Country={v}" for v in row["Country"]]
            + ([f"Rating={row['rating']}"] if pd.notna(row["rating"]) else [])
            + ([f"Type={row['type']}"] if pd.notna(row["type"]) else [])
        )
        return list(dict.fromkeys(transaction))

    df["transaction_items"] = df.apply(items, axis=1)
    return df.reset_index(drop=True)


def build_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genre = (df[["show_id", "title", "Genre"]].explode("Genre")
             .rename(columns={"Genre": "genre"}).dropna(subset=["genre"]))
    genre = genre[genre["genre"] != ""].reset_index(drop=True)
    country = (df[["show_id", "title", "Country"]].explode("Country")
               .rename(columns={"Country": "country"}).dropna(subset=["country"]))
    country = country[country["country"] != ""].reset_index(drop=True)
    transactions = df[["show_id", "title", "transaction_items"]].copy()

    # Cùng nguyên lý One-Hot Encoding như TransactionEncoder trong notebook.
    encoder = MultiLabelBinarizer()
    matrix = encoder.fit_transform(df["transaction_items"].tolist())
    basket = pd.DataFrame(matrix.astype(bool), columns=encoder.classes_, index=df["show_id"].astype(str))
    basket.index.name = "show_id"
    return genre, country, transactions, basket


def mine_apriori(basket: pd.DataFrame, min_support: float = 0.03,
                 min_confidence: float = 0.50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frequent Itemsets -> Association Rules (mlxtend), như notebook."""
    from mlxtend.frequent_patterns import apriori, association_rules
    if basket.shape[0] == 0 or basket.shape[1] == 0:
        return pd.DataFrame(columns=["support", "itemsets"]), pd.DataFrame()
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if frequent.empty or frequent["itemsets"].map(len).max() < 2:
        return frequent, pd.DataFrame()
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    return frequent, rules


def fit_kmeans_pca(basket: pd.DataFrame, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-Means với random_state=42, n_init=10; PCA 2 chiều như notebook."""
    x = basket.to_numpy(dtype=np.float32, copy=True)
    if len(x) < k:
        raise ValueError(f"Chỉ có {len(x)} dòng, không thể gom thành {k} cụm")
    if x.shape[1] < 2:
        raise ValueError("Không đủ ít nhất 2 đặc trưng để chạy PCA")
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(x)
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(x)
    return labels, coordinates, pca.explained_variance_ratio_


def format_itemset(items: object) -> str:
    if isinstance(items, (set, frozenset, tuple, list)):
        return " + ".join(sorted(str(x) for x in items))
    return str(items)


def export_csv(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8-sig")


def matching_titles(basket: pd.DataFrame, show_id: str, limit: int = 8,
                    allowed_ids: set[str] | None = None) -> pd.DataFrame:
    """Độ tương đồng Jaccard trên các đặc trưng nhị phân, KHÔNG là lịch sử xem."""
    if show_id not in basket.index:
        raise ValueError("Không tìm thấy show_id đã chọn")
    x = basket.to_numpy(dtype=np.bool_, copy=False)
    selected = basket.index.get_loc(show_id)
    query = x[selected]
    intersection = np.logical_and(x, query).sum(axis=1)
    union = np.logical_or(x, query).sum(axis=1)
    similarity = np.divide(intersection, union, out=np.zeros(len(x), dtype=float), where=union > 0)
    similarity[selected] = -1
    if allowed_ids is not None:
        valid = np.asarray(basket.index.isin(allowed_ids), dtype=bool)
        similarity[~valid] = -1
    inds = np.argsort(-similarity, kind="stable")
    inds = inds[similarity[inds] > 0][:limit]
    return pd.DataFrame({"show_id": basket.index[inds], "similarity": similarity[inds]})
