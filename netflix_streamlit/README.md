# 🎬 Netflix Data Mining — Demo Streamlit

Ứng dụng được chuyển từ notebook **`đồ_án_netflix.ipynb`**. Giữ các bước xử lý dữ liệu, **Apriori** (mặc định Support 0.03, Confidence 0.50), **K-Means** (mặc định K=5, random_state=42, n_init=10) và **PCA** 2 chiều.

## Chuẩn bị dataset

Tải tại: https://www.kaggle.com/datasets/shivamb/netflix-shows

Dữ liệu cần dùng: `netflix_titles.csv` (có thể để nguyên ZIP). Không đóng gói dữ liệu Kaggle vào bộ code. Chọn **một** trong các cách:
1. Đặt `netflix_titles.csv` vào thư mục `data/` cùng project.
2. Mở ứng dụng, **Upload** file `netflix_titles.csv` / `.zip` ở sidebar.
3. Mở ứng dụng, nhấn **Tải dataset từ Kaggle** (cần Internet lần đầu; một số môi trường yêu cầu đăng nhập/tải thủ công).

## Chạy trên Windows (VS Code)

Cài Python 3.10 trở lên (khuyến nghị Python 3.11 hoặc 3.12), mở Terminal trong đúng thư mục project:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Hoặc **nháy đúp** `CHAY_STREAMLIT_WINDOWS.bat` để cài thư viện và mở Streamlit.

Mở địa chỉ mà Terminal hiển thị (thường là `http://localhost:8501`).

## Các trang trong demo

| Trang | Bạn có thể làm gì? |
|---|---|
| 📊 Tổng quan Netflix | Xem số tựa phim, movie, TV show, thể loại, quốc gia và biểu đồ. |
| 🧹 Tiền xử lý | So sánh giá trị thiếu, kiểm tra bảng Genre/Country, transaction và basket; tải file CSV. |
| 🧩 Thuật toán Apriori | Điều chỉnh Support/Confidence, xem frequent itemsets, luật kết hợp theo Lift và tải CSV. |
| 🔵 Gom cụm K-Means | Thay đổi K, xem các cụm trên biểu đồ PCA, phân tích đặc trưng nổi bật và tải dữ liệu đã gắn cluster. |
| 🎞️ Khám phá nội dung | Tìm tên phim, xem mô tả, đề xuất nội dung tương đồng bằng Jaccard, tùy chọn lọc trong cùng cụm. |
| 📚 Quy trình & giải thích | Xem luồng xử lý, dữ liệu đầu vào và kết quả đầu ra của từng bước. |

### Cấu trúc thư mục

```text
netflix_streamlit/
├── app.py
├── netflix_core.py
├── requirements.txt
├── CHAY_STREAMLIT_WINDOWS.bat
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   ├── README.txt
│   └── netflix_titles.csv   # bạn tải từ Kaggle, không có sẵn
└── tests/
    └── test_core.py
```

## Giải thích nhanh thuật toán

- **Transaction:** mỗi tựa phim là một giao dịch, ví dụ `Genre=Dramas`, `Country=India`, `Rating=TV-MA`, `Type=MOVIE`.
- **Support:** tỷ lệ tiêu đề có đủ các thuộc tính trong luật.
- **Confidence:** trong các tiêu đề có vế trái, tỷ lệ có cả vế phải.
- **Lift:** mức đồng xuất hiện so với giả định hai vế độc lập.
- **K-Means:** gom thành K cụm theo vector One-hot; số hiệu cụm chỉ là nhãn, không mang ý nghĩa cố định.
- **PCA:** giảm số chiều giúp trực quan 2D; K-Means vẫn được huấn luyện bằng đủ cột One-hot.
- **Tương đồng Jaccard:** số đặc trưng cùng xuất hiện chia cho số đặc trưng xuất hiện ở ít nhất một trong hai nội dung.

> **Giới hạn:** Kaggle dataset này là ảnh chụp danh mục Netflix trong quá khứ, không phải dữ liệu Netflix thời gian thực. Không có lượt xem hay đánh giá theo từng người nên không thể kết luận về sở thích người dùng hoặc hành vi xem từ những luật kết hợp này.

## Chạy kiểm thử (tùy chọn)

```powershell
python -m pip install pytest
python -m pytest tests -q
```
