"""Netflix Data Mining Dashboard — Streamlit demo từ notebook đồ_án_netflix.ipynb.

Chạy: python -m streamlit run app.py
Dataset gốc: https://www.kaggle.com/datasets/shivamb/netflix-shows
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from netflix_core import (
    build_tables, clean_netflix, export_csv, fit_kmeans_pca,
    format_itemset, matching_titles, mine_apriori, read_netflix_csv,
)

ROOT = Path(__file__).resolve().parent
KAGGLE_URL = "https://www.kaggle.com/datasets/shivamb/netflix-shows"

st.set_page_config(
    page_title="Netflix Insights | Apriori & K-Means",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background: linear-gradient(160deg, #101116 0%, #0b0d12 55%, #111017 100%); }
  [data-testid="stSidebar"] { background: #14161d; border-right: 1px solid #282a33; }
  [data-testid="stMetric"] { background: #191c25; border: 1px solid #2a2d37;
     border-radius: 14px; padding: 16px 20px; }
  [data-testid="stMetricLabel"] { color: #b8bdc9; }
  [data-testid="stMetricValue"] { color: #f6f7fa; }
  .block-container { padding-top: 1.6rem; max-width: 1450px; }
  .hero { background: linear-gradient(110deg, rgba(229,9,20,.20), rgba(23,26,34,.92) 48%);
    border: 1px solid rgba(229,9,20,.34); border-radius: 19px;
    padding: 24px 28px; margin-bottom: 22px; }
  .hero h1 { margin: 0; color: #fff; font-size: 2.35rem; letter-spacing: -.04em; }
  .hero p { margin: 11px 0 0 0; font-size: 1rem; color: #d1d4df; }
  .eyebrow { color: #ff5660; font-size: .82rem; font-weight: 800; letter-spacing: .14em; }
  .info-panel { border: 1px solid #32333e; background: #191b24; border-radius: 12px;
    padding: 18px 20px; margin: 12px 0 22px 0; color: #d7d9e2; }
  .pill { display: inline-block; background: #3e1720; color: #ff777f;
    padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  a { color: #ff606b; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def prepare(contents: bytes, filename: str):
    raw = read_netflix_csv(contents, filename)
    clean = clean_netflix(raw)
    genre, country, transactions, basket = build_tables(clean)
    return raw, clean, genre, country, transactions, basket


@st.cache_data(show_spinner="Đang chạy thuật toán Apriori và tính luật kết hợp...")
def cached_apriori(basket: pd.DataFrame, support: float, confidence: float):
    return mine_apriori(basket, support, confidence)


@st.cache_data(show_spinner="Đang huấn luyện K-Means và chiếu dữ liệu bằng PCA...")
def cached_kmeans(basket: pd.DataFrame, k: int):
    return fit_kmeans_pca(basket, k)


def plot_defaults(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=20, t=26, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e2eb"), legend_title_text="",
    )
    return fig


def download_button(label: str, frame: pd.DataFrame, filename: str,
                    key: str, index: bool = False):
    st.download_button(
        label, data=export_csv(frame, index=index), file_name=filename,
        mime="text/csv", key=key, use_container_width=True,
    )


def header(section: str, lead: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">NETFLIX DATA MINING / {escape(section)}</div>'
                f'<h1>{escape(lead)}</h1>'
                '<p>Phân tích dữ liệu Netflix bằng Apriori, K-Means và PCA • Từ notebook sang ứng dụng tương tác</p>'
                '</div>', unsafe_allow_html=True)


def local_dataset_path() -> Path | None:
    candidates = (
        ROOT / "data" / "netflix_titles.csv",
        ROOT / "data" / "netflix_titles.csv.zip",
        ROOT / "data" / "netflix-shows.zip",
        ROOT / "netflix_titles.csv",
        ROOT / "netflix_titles.csv.zip",
    )
    return next((p for p in candidates if p.is_file()), None)


def download_kaggle() -> Path:
    """Kaggle public dataset; người dùng cần kết nối mạng ở lần tải đầu."""
    import kagglehub
    dataset_dir = Path(kagglehub.dataset_download("shivamb/netflix-shows"))
    matches = list(dataset_dir.rglob("netflix_titles.csv"))
    if not matches:
        raise FileNotFoundError("Kaggle đã tải nhưng không tìm thấy netflix_titles.csv")
    return matches[0]


with st.sidebar:
    st.markdown("## 🎬 NETFLIX INSIGHTS")
    st.caption("Bài tập khai phá dữ liệu • Streamlit")
    st.divider()
    uploaded = st.file_uploader("📂 Upload dữ liệu Kaggle", type=["csv", "zip"],
                                help="Hỗ trợ netflix_titles.csv hoặc ZIP chứa file CSV")
    if st.button("☁️ Tải dataset từ Kaggle", use_container_width=True):
        with st.spinner("Đang lấy Netflix Movies and TV Shows từ Kaggle..."):
            try:
                st.session_state["kaggle_file"] = str(download_kaggle())
                st.success("Đã tải dataset từ Kaggle!")
            except Exception as exc:
                st.error(f"Không tải tự động được: {exc}")
                st.info("Có thể tải thủ công từ trang Kaggle rồi upload file CSV/ZIP.")
    st.caption(f"[Nguồn dữ liệu Kaggle]({KAGGLE_URL})")
    st.divider()
    nav = st.radio("Chọn chức năng", [
        "📊 Tổng quan Netflix",
        "🧹 Tiền xử lý dữ liệu",
        "🧩 Thuật toán Apriori",
        "🔵 Gom cụm K-Means",
        "🎞️ Khám phá nội dung",
        "📚 Quy trình & giải thích",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("⚠️ Bộ dữ liệu là ảnh chụp danh mục Netflix lịch sử, không phản ánh thư viện Netflix trực tiếp năm 2026.")

contents = None
filename = None
source_label = None
if uploaded is not None:
    contents = uploaded.getvalue()
    filename = uploaded.name
    source_label = f"File upload: {uploaded.name}"
else:
    kaggle_file = st.session_state.get("kaggle_file")
    p = Path(kaggle_file) if kaggle_file else local_dataset_path()
    if p is not None and p.is_file():
        contents, filename, source_label = p.read_bytes(), p.name, f"Đã đọc: {p.name}"

if contents is None:
    header("GET STARTED", "Khai phá dữ liệu Netflix")
    st.info("**Cần file dữ liệu để bắt đầu.** Upload `netflix_titles.csv` / `.zip` ở sidebar, "
            "nhấn **Tải dataset từ Kaggle**, hoặc đặt `netflix_titles.csv` trong thư mục `data/`.")
    st.markdown("""
    ### App bao gồm gì?
    - **Dashboard:** tổng số phim, TV Show, thể loại, quốc gia và xu hướng theo năm.
    - **Tiền xử lý:** dữ liệu gốc, báo cáo thiếu dữ liệu, tách Genre/Country, one-hot transaction.
    - **Apriori:** khai phá Frequent Itemsets và luật kết hợp với Support / Confidence / Lift.
    - **K-Means + PCA:** gom các tựa phim thành cụm và vẽ biểu đồ 2D.
    - **Khám phá:** tìm phim và gợi ý phim tương đồng dựa trên thuộc tính metadata.
    """)
    st.stop()

try:
    raw, df, genre_df, country_df, transactions_df, basket = prepare(contents, filename)
except Exception as exc:
    st.error(f"Dữ liệu không hợp lệ: {exc}")
    st.stop()

if df.empty:
    st.error("Dữ liệu không còn bản ghi hợp lệ sau khi làm sạch. Hãy kiểm tra lại show_id và các cột bắt buộc.")
    st.stop()

with st.sidebar:
    st.success("✅ Đã nạp dữ liệu")
    st.caption(source_label)
    st.caption(f"{len(df):,} tựa phim/chương trình • {basket.shape[1]} đặc trưng One-hot")

# --- DASHBOARD ---
if nav == "📊 Tổng quan Netflix":
    header("DASHBOARD", "Netflix Data Explorer")
    movies = int((df["type"] == "MOVIE").sum())
    tvshows = int((df["type"] == "TV SHOW").sum())
    stat_cols = st.columns(5)
    metrics = [
        ("Tổng nội dung", f"{len(df):,}"),
        ("Phim điện ảnh", f"{movies:,}"),
        ("TV Shows", f"{tvshows:,}"),
        ("Thể loại", f"{genre_df['genre'].nunique():,}"),
        ("Quốc gia", f"{country_df['country'].nunique():,}"),
    ]
    for col, (label, value) in zip(stat_cols, metrics):
        with col:
            st.metric(label, value)

    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("Loại nội dung")
        types = df["type"].fillna("KHÔNG RÕ").value_counts().reset_index()
        types.columns = ["Loại", "Số lượng"]
        fig = px.pie(types, names="Loại", values="Số lượng", hole=.64,
                     color_discrete_sequence=["#e50914", "#527af4", "#b3b7c6"])
        fig.update_traces(textinfo="percent+label", textposition="outside")
        st.plotly_chart(plot_defaults(fig), use_container_width=True)
    with right:
        st.subheader("Phân bố theo năm phát hành")
        years = pd.to_numeric(df["release_year"], errors="coerce")
        year_counts = years.dropna().astype(int).value_counts().sort_index()
        fig = px.area(x=year_counts.index, y=year_counts.values,
                      labels={"x": "Năm phát hành", "y": "Số tựa phim"})
        fig.update_traces(line_color="#ef3141", fillcolor="rgba(229,9,20,.30)")
        st.plotly_chart(plot_defaults(fig), use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Top 12 thể loại")
        top_genre = genre_df["genre"].value_counts().head(12).sort_values()
        fig = px.bar(x=top_genre.values, y=top_genre.index, orientation="h",
                     labels={"x": "Số nội dung", "y": "Thể loại"})
        fig.update_traces(marker_color="#e50914")
        st.plotly_chart(plot_defaults(fig, 430), use_container_width=True)
    with c2:
        st.subheader("Top 12 quốc gia")
        top_country = country_df["country"].value_counts().head(12).sort_values()
        fig = px.bar(x=top_country.values, y=top_country.index, orientation="h",
                     labels={"x": "Số nội dung", "y": "Quốc gia"})
        fig.update_traces(marker_color="#6579eb")
        st.plotly_chart(plot_defaults(fig, 430), use_container_width=True)

    st.subheader("Xem nhanh dữ liệu")
    st.dataframe(df[["show_id", "title", "type", "country", "release_year", "rating", "listed_in"]].head(30),
                 hide_index=True, use_container_width=True)
    st.caption("Quốc gia và thể loại được tách riêng; một phim có thể xuất hiện trong nhiều nhóm nên tổng nhóm lớn hơn tổng số phim.")

# --- PREPROCESS ---
elif nav == "🧹 Tiền xử lý dữ liệu":
    header("PREPROCESSING", "Làm sạch & One-Hot Encoding")
    st.markdown("**Quy trình như notebook:** Chuẩn hóa tên cột → bỏ khoảng trắng → chuyển chuỗi rỗng thành NaN → "
                "xóa trùng → chuẩn hóa Type/Rating → chuyển Date → tách Genre/Country → transaction → One-hot.")
    a, b, c = st.columns(3)
    a.metric("Dữ liệu ban đầu", f"{len(raw):,} dòng")
    b.metric("Sau khi làm sạch", f"{len(df):,} dòng")
    c.metric("Số đặc trưng One-hot", f"{basket.shape[1]:,}")

    st.subheader("1. Báo cáo giá trị thiếu")
    raw_cols = raw.copy()
    raw_cols.columns = raw_cols.columns.str.strip().str.lower()
    missing = pd.DataFrame({
        "Cột": df.drop(columns=["Genre", "Country", "transaction_items"]).columns,
    })
    missing["Thiếu trước"] = missing["Cột"].map(raw_cols.isna().sum()).fillna(0).astype(int)
    missing["Thiếu sau"] = missing["Cột"].map(df.isna().sum()).fillna(0).astype(int)
    missing["Tỷ lệ thiếu sau (%)"] = (missing["Thiếu sau"] / len(df) * 100).round(2)
    st.dataframe(missing, hide_index=True, use_container_width=True)
    st.caption("Notebook chỉ loại bản ghi trùng và show_id thiếu; các trường country/rating thiếu vẫn được giữ để tránh mất dữ liệu.")

    st.subheader("2. Chuyển thể loại và quốc gia thành nhiều dòng")
    tab_g, tab_c = st.tabs(["Genre", "Country"])
    with tab_g:
        st.dataframe(genre_df.head(30), hide_index=True, use_container_width=True)
        download_button("⬇️ Tải netflix_genre.csv", genre_df, "netflix_genre.csv", "dl_genre")
    with tab_c:
        st.dataframe(country_df.head(30), hide_index=True, use_container_width=True)
        download_button("⬇️ Tải netflix_country.csv", country_df, "netflix_country.csv", "dl_country")

    st.subheader("3. Giao dịch (transactions) của từng tựa phim")
    tx_disp = transactions_df.head(15).copy()
    tx_disp["transaction_items"] = tx_disp["transaction_items"].apply(lambda x: ", ".join(x))
    st.dataframe(tx_disp, hide_index=True, use_container_width=True)
    st.caption("Ví dụ transaction: Genre=Dramas, Country=United States, Rating=TV-MA, Type=MOVIE")

    st.subheader("4. Bảng One-hot Encoding")
    preview_cols = list(basket.columns[:18])
    st.dataframe(basket[preview_cols].head(12).astype(int), use_container_width=True)
    st.caption("1 = tựa phim có thuộc tính này; 0 = không có. Bảng thực tế có đủ các cột; trên màn hình chỉ xem 18 cột đầu.")
    d1, d2, d3 = st.columns(3)
    with d1:
        clean_export = df.drop(columns=["Genre", "Country", "transaction_items"])
        download_button("⬇️ Dữ liệu đã làm sạch", clean_export, "netflix_clean.csv", "dl_clean")
    with d2:
        tx_export = transactions_df.copy()
        tx_export["transaction_items"] = tx_export["transaction_items"].apply(lambda x: " | ".join(x))
        download_button("⬇️ Bảng transactions", tx_export, "netflix_transactions.csv", "dl_tx")
    with d3:
        download_button("⬇️ Bảng basket", basket.astype(int), "netflix_basket.csv", "dl_basket", index=True)

# --- APRIORI ---
elif nav == "🧩 Thuật toán Apriori":
    header("ASSOCIATION RULES", "Apriori • Luật kết hợp")
    st.markdown("**Mục tiêu:** Tìm những thuộc tính metadata thường xuất hiện cùng nhau trong danh mục Netflix "
                "(thể loại, quốc gia, độ tuổi, loại phim). **Không** phải luật về hành vi người xem.")
    sc1, sc2 = st.columns(2)
    with sc1:
        min_support = st.slider("Minimum Support", min_value=.02, max_value=.15,
                                value=.03, step=.01, format="%.2f",
                                help="Tỷ lệ tối thiểu các tựa phim chứa tập mục.")
    with sc2:
        min_confidence = st.slider("Minimum Confidence", min_value=.30, max_value=.95,
                                   value=.50, step=.05, format="%.2f",
                                   help="Tỷ lệ tựa phim có B trong nhóm có A.")
    try:
        frequent, rules = cached_apriori(basket, min_support, min_confidence)
    except ImportError:
        st.error("Chưa cài mlxtend. Chạy `python -m pip install -r requirements.txt` rồi mở lại app.")
        st.stop()
    except Exception as exc:
        st.error(f"Không chạy Apriori được: {exc}")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Số giao dịch", f"{len(basket):,}")
    m2.metric("Frequent Itemsets", f"{len(frequent):,}")
    m3.metric("Luật kết hợp", f"{len(rules):,}")

    st.subheader("1. Frequent Itemsets")
    frequent_table = frequent.copy()
    if not frequent_table.empty:
        frequent_table["length"] = frequent_table["itemsets"].apply(len)
        frequent_table["itemsets"] = frequent_table["itemsets"].apply(format_itemset)
        frequent_table = frequent_table.sort_values("support", ascending=False).reset_index(drop=True)
        frequent_table = frequent_table.rename(columns={"itemsets": "Tập mục", "support": "Support", "length": "Số item"})
        st.dataframe(frequent_table.head(60), hide_index=True, use_container_width=True,
                     column_config={"Support": st.column_config.ProgressColumn("Support", format="%.3f", min_value=0, max_value=1)})
        download_button("⬇️ Tải frequent_itemsets.csv", frequent_table,
                        "netflix_frequent_itemsets.csv", "dl_itemsets")
    else:
        st.warning("Không có tập mục đạt ngưỡng Support; hãy thử giảm ngưỡng.")

    st.subheader("2. Association Rules — ưu tiên Lift cao")
    if rules.empty:
        st.warning("Không tìm thấy luật thỏa Confidence và Support đã chọn. Hãy thử giảm ngưỡng.")
    else:
        rules_table = rules.copy()
        rules_table["antecedents"] = rules_table["antecedents"].apply(format_itemset)
        rules_table["consequents"] = rules_table["consequents"].apply(format_itemset)
        view = rules_table[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
        view.columns = ["Nếu có (A)", "Thường đi kèm (B)", "Support", "Confidence", "Lift"]
        st.dataframe(view.head(100), hide_index=True, use_container_width=True,
                     column_config={
                         "Support": st.column_config.NumberColumn(format="%.3f"),
                         "Confidence": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=1),
                         "Lift": st.column_config.NumberColumn(format="%.2f"),
                     })
        download_button("⬇️ Tải association_rules.csv", rules_table,
                        "netflix_association_rules.csv", "dl_rules")

        best = view.iloc[0]
        st.markdown(
            f'<div class="info-panel"><span class="pill">TOP RULE BY LIFT</span><p>'
            f'<b>Nếu nội dung có:</b> {escape(best["Nếu có (A)"])}<br>'
            f'<b>Thường đi kèm:</b> {escape(best["Thường đi kèm (B)"])}<br>'
            f'Confidence = {best["Confidence"]:.1%} · Support = {best["Support"]:.1%} · Lift = {best["Lift"]:.2f}'
            '</p></div>', unsafe_allow_html=True)

        st.subheader("3. Trực quan luật kết hợp")
        fig = px.scatter(view.head(200), x="Support", y="Confidence", size="Lift", color="Lift",
                         hover_data=["Nếu có (A)", "Thường đi kèm (B)"],
                         color_continuous_scale="Reds", size_max=26)
        fig.update_layout(coloraxis_colorbar_title="Lift")
        st.plotly_chart(plot_defaults(fig, 450), use_container_width=True)
    with st.expander("📘 Giải thích Support, Confidence, Lift để thuyết trình"):
        st.markdown(r"""
        - **Support(A → B):** tỷ lệ toàn bộ phim có **cả A và B**.
        - **Confidence(A → B):** trong nhóm phim có **A**, tỷ lệ cũng có **B**.
        - **Lift(A → B):** so sánh mức đồng xuất hiện của A và B với mức kỳ vọng nếu độc lập.
          **Lift > 1**: đồng xuất hiện nhiều hơn mức kỳ vọng; **Lift = 1**: gần mức độc lập.
        - Ví dụ: `Genre=Dramas → Type=MOVIE` cho biết trong các tựa phim gắn nhãn Dramas,
          bao nhiêu phần trăm thuộc loại Movie. Đây là **liên hệ trong dữ liệu**, không chứng minh nhân quả.
        """)

# --- KMEANS ---
elif nav == "🔵 Gom cụm K-Means":
    header("CLUSTERING", "K-Means • Gom nhóm nội dung")
    st.markdown("**Ý tưởng:** mỗi tựa phim là vector 0/1 gồm Genre, Country, Rating và Type. "
                "K-Means gom những tựa phim có đặc trưng tương tự; PCA giúp xem cụm trong không gian 2D.")
    if len(df) < 2:
        st.warning("Cần tối thiểu 2 tựa phim để gom cụm.")
        st.stop()
    k = st.slider("Chọn số cụm K", 2, min(10, len(df)), min(5, len(df)),
                  help="Notebook ban đầu chọn K=5. Thay đổi K để so sánh kết quả.")
    try:
        labels, coords, ratios = cached_kmeans(basket, k)
    except Exception as exc:
        st.error(f"Không tính được K-Means/PCA: {exc}")
        st.stop()

    plot_df = pd.DataFrame({
        "PCA 1": coords[:, 0], "PCA 2": coords[:, 1],
        "Cụm": [f"Cluster {i}" for i in labels],
        "title": df["title"].fillna("Không rõ"),
        "type": df["type"].fillna("Không rõ"),
        "release_year": df["release_year"],
        "show_id": df["show_id"].astype(str),
    })
    c1, c2, c3 = st.columns(3)
    c1.metric("Số cụm", k)
    c2.metric("Số nội dung đã gom", f"{len(plot_df):,}")
    c3.metric("Phương sai 2 thành phần PCA", f"{sum(ratios):.1%}")
    st.caption("PCA là cách giảm chiều để vẽ hình. Khoảng cách trên hình 2D có thể không giữ nguyên mọi quan hệ từ dữ liệu gốc.")

    max_points = len(plot_df)
    if len(plot_df) > 2000:
        nplot = st.slider("Số điểm hiển thị trên biểu đồ", 1000, len(plot_df),
                          min(3000, len(plot_df)), step=100)
    else:
        nplot = max_points
    if nplot < max_points:
        sampled = plot_df.groupby("Cụm", group_keys=False).sample(
            frac=nplot / max_points, random_state=42).head(nplot)
    else:
        sampled = plot_df
    fig = px.scatter(sampled, x="PCA 1", y="PCA 2", color="Cụm", hover_name="title",
                     hover_data={"type": True, "release_year": True, "show_id": True, "Cụm": False},
                     opacity=.68, color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_traces(marker=dict(size=6))
    st.plotly_chart(plot_defaults(fig, 570), use_container_width=True)

    st.subheader("Số lượng và đặc trưng nổi bật theo cụm")
    global_prevalence = basket.mean(axis=0)
    cluster_rows = []
    for cluster in range(k):
        mask = labels == cluster
        count = int(np.sum(mask))
        features = basket.iloc[np.flatnonzero(mask)]
        if count:
            difference = (features.mean(axis=0) - global_prevalence)
            best_features = difference.sort_values(ascending=False).head(4).index.tolist()
            examples = df.loc[mask, "title"].dropna().astype(str).head(3).tolist()
        else:
            best_features, examples = [], []
        cluster_rows.append({
            "Cụm": f"Cluster {cluster}", "Số tựa phim": count,
            "Đặc trưng nổi bật (so với toàn bộ)": " • ".join(best_features),
            "Ví dụ nội dung": " • ".join(examples),
        })
    cluster_table = pd.DataFrame(cluster_rows)
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        counts = plot_df["Cụm"].value_counts().sort_index()
        fig = px.bar(x=counts.index, y=counts.values, labels={"x": "Cụm", "y": "Số nội dung"},
                     color=counts.index, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False)
        st.plotly_chart(plot_defaults(fig, 385), use_container_width=True)
    with right:
        st.dataframe(cluster_table, hide_index=True, use_container_width=True, height=360)
    st.caption("Nhãn Cluster 0, 1, 2... do thuật toán tự đặt. 'Đặc trưng nổi bật' là thuộc tính phổ biến hơn mức trung bình toàn bộ dữ liệu; không phải nhãn thể loại cố định.")

    df_clustered = df.drop(columns=["Genre", "Country", "transaction_items"]).copy()
    df_clustered["cluster"] = labels
    download_button("⬇️ Tải netflix_clean_clustered.csv", df_clustered,
                    "netflix_clean_clustered.csv", "dl_cluster")
    with st.expander("📘 Gom cụm xong để làm gì?"):
        st.markdown("""
        K-Means giúp **chia danh mục Netflix thành các nhóm nội dung gần giống nhau** về thể loại, quốc gia, mức phân loại độ tuổi và Movie/TV Show.
        Có thể dùng các nhóm để **khám phá cấu trúc danh mục**, xem mỗi nhóm nổi bật gì, hoặc thu hẹp phạm vi tìm những phim có metadata tương tự.
        Gom cụm **không tự chứng minh người xem thích phim nào** vì dataset không có dữ liệu người dùng, lượt xem hay đánh giá cá nhân.
        """)

# --- EXPLORER ---
elif nav == "🎞️ Khám phá nội dung":
    header("CONTENT EXPLORER", "Tìm phim & nội dung tương tự")
    st.markdown("Tìm một tựa phim có trong dataset và xem những nội dung **tương đồng về metadata**. "
                "Chức năng này bổ sung cho notebook để demo kết quả cụm và bảng One-hot.")
    lc, rc = st.columns([2, 1])
    with lc:
        keyword = st.text_input("Tìm tên phim hoặc TV Show", placeholder="Ví dụ: Stranger Things, Queen's Gambit...")
    with rc:
        type_filter = st.selectbox("Loại nội dung", ["Tất cả", "MOVIE", "TV SHOW"])
    filtered = df
    if keyword.strip():
        filtered = filtered[filtered["title"].fillna("").astype(str).str.contains(keyword.strip(), regex=False, case=False)]
    if type_filter != "Tất cả":
        filtered = filtered[filtered["type"] == type_filter]
    st.caption(f"Tìm thấy {len(filtered):,} tựa phim/chương trình")
    if filtered.empty:
        st.warning("Không tìm thấy nội dung phù hợp. Hãy thử từ khóa khác.")
        st.stop()
    label_lookup = {
        str(row.show_id): f"{row.title} ({row.release_year}) • {row.show_id}"
        for row in filtered[["show_id", "title", "release_year"]].itertuples(index=False)
    }
    selected_id = st.selectbox("Chọn nội dung", options=list(label_lookup),
                               format_func=lambda sid: label_lookup[sid])
    selected = df[df["show_id"].astype(str) == selected_id].iloc[0]
    st.markdown(f"### 🎞️ {escape(str(selected['title']))}")
    st.write(f"**Loại:** {selected['type']}   |   **Năm:** {selected['release_year']}   |   "
             f"**Rating:** {selected['rating'] if pd.notna(selected['rating']) else 'Không rõ'}")
    st.write(f"**Thể loại:** {', '.join(selected['Genre']) or 'Không rõ'}")
    st.write(f"**Quốc gia:** {', '.join(selected['Country']) or 'Không rõ'}")
    if "description" in df.columns and pd.notna(selected.get("description")):
        st.write(f"**Mô tả:** {selected['description']}")
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        only_cluster = st.checkbox("Chỉ tìm phim trong cùng cụm K-Means (K=5)", value=True)
    with right:
        limit = st.slider("Số nội dung tương tự", 3, 15, 8)
    allowed_ids = None
    if only_cluster:
        if len(df) >= 5:
            labels, _, _ = cached_kmeans(basket, 5)
            chosen_index = basket.index.get_loc(selected_id)
            chosen_cluster = labels[chosen_index]
            allowed_ids = set(basket.index[labels == chosen_cluster])
            st.caption(f"Đang tìm trong Cluster {chosen_cluster} ({len(allowed_ids):,} nội dung).")
        else:
            st.info("Dữ liệu dưới 5 nội dung nên bỏ qua điều kiện K-Means K=5.")
    suggestions = matching_titles(basket, selected_id, limit=limit, allowed_ids=allowed_ids)
    if suggestions.empty:
        st.info("Không tìm thấy phim trùng ít nhất một thuộc tính trong phạm vi đã chọn.")
    else:
        display = suggestions.merge(df[["show_id", "title", "type", "release_year", "listed_in"]],
                                    on="show_id", how="left")
        display["Độ tương đồng (%)"] = (display.pop("similarity") * 100).round(1)
        display = display.rename(columns={"title": "Tên nội dung", "type": "Loại",
                                          "release_year": "Năm", "listed_in": "Thể loại"})
        st.dataframe(display[["Tên nội dung", "Loại", "Năm", "Thể loại", "Độ tương đồng (%)"]],
                     hide_index=True, use_container_width=True)
        st.caption("Độ tương đồng dùng Jaccard trên các mục Genre/Country/Rating/Type (0–100%). "
                   "Đây là độ giống thuộc tính, không phải xác suất người xem thích phim.")

# --- PROCESS ---
else:
    header("METHODOLOGY", "Pipeline khai phá dữ liệu")
    st.markdown("""
    ### Luồng xử lý dữ liệu

    **1. Netflix CSV → 2. Làm sạch → 3. Tách Genre/Country → 4. Tạo transaction → 5. One-hot Encoding**

    Từ bước 5, cùng một bảng basket được đưa tới hai hướng:

    - **Apriori:** tìm tập mục phổ biến → sinh luật kết hợp → đánh giá Support, Confidence, Lift.
    - **K-Means:** gom cụm dựa trên vector thuộc tính → phân tích cụm → dùng PCA giảm xuống 2 chiều để vẽ.

    ### Input / Output trong notebook
    | Bước | Input | Output |
    |---|---|---|
    | Làm sạch | `netflix_titles.csv` | `netflix_clean.csv` |
    | Tách danh sách | `listed_in`, `country` | `netflix_genre.csv`, `netflix_country.csv` |
    | Transaction | Genre, Country, Rating, Type | `netflix_transactions.csv` |
    | One-hot | Transactions | `netflix_basket.csv` |
    | Apriori | Basket, support 0.03, confidence 0.5 | `netflix_frequent_itemsets.csv`, `netflix_association_rules.csv` |
    | K-Means | Basket, K=5 | `netflix_clean_clustered.csv` và biểu đồ PCA |

    """)
    st.info("**Lưu ý khoa học:** Đây là bài toán khai phá thuộc tính trong danh mục nội dung, không phải dữ liệu hành vi xem. "
            "Không nên diễn giải các luật Apriori là 'người xem phim A sẽ xem phim B'.")
    st.markdown(f"Nguồn dataset: [Netflix Movies and TV Shows — Kaggle]({KAGGLE_URL})")
