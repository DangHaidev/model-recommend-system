    movielens-recommender/
    ├── data/
    │ ├── raw/
    │ └── processed/
    ├── notebooks/
    │ └── EDA.ipynb
    ├── src/
    │ ├── data_loader.py
    │ ├── content_based.py
    │ ├── collaborative.py
    │ ├── hybrid.py
    │ ├── evaluation.py
    │ └── utils.py
    ├── app/
    │ ├── api.py
    │ └── ui.py
    ├── tests/
    ├── requirements.txt
    ├── README.md
    └── main.py

📁 data/
data/raw/

Chứa dữ liệu gốc, chưa qua xử lý, ví dụ: các file .csv từ MovieLens (ratings.csv, movies.csv, v.v.)

Không nên chỉnh sửa thủ công: dùng script để xử lý.

data/processed/

Lưu trữ dữ liệu đã xử lý (đã encode, đã lọc, clean...), có thể dưới dạng .csv, .pkl, .parquet

Mục tiêu: giúp bạn không phải xử lý lại dữ liệu mỗi lần chạy

✅ Tách biệt dữ liệu thô và xử lý giúp bạn dễ kiểm soát quy trình tiền xử lý, dễ tái tạo kết quả.

📁 notebooks/
EDA.ipynb

Notebook phân tích dữ liệu ban đầu (Exploratory Data Analysis)

Hiển thị:

Thống kê user, movie

Phân phối rating

Heatmap, histogram, v.v.

✅ Dùng để hiểu rõ dữ liệu trước khi xây mô hình.

📁 src/ — MÃ NGUỒN CHÍNH
data_loader.py

Đọc dữ liệu từ data/raw/

Có thể xử lý nhẹ (loại bỏ null, merge bảng...)

Hàm chính: load_data()

content_based.py

Gợi ý phim dựa trên nội dung phim (genre, tag, etc.)

Dùng TF-IDF, cosine_similarity, etc.

Hàm chính: recommend_by_content(user_id) hoặc get_similar_movies(movie_id)

collaborative.py

Mô hình collaborative filtering (user-user, item-item, SVD, ALS)

Có thể dùng thư viện như Surprise, LightFM, implicit

Hàm chính: recommend_by_cf(user_id)

hybrid.py

Kết hợp cả hai mô hình trên

Có thể dùng simple weighted average hoặc mô hình học máy (XGBoost, Neural CF...)

Hàm chính: recommend_hybrid(user_id)

evaluation.py

Chứa hàm đánh giá mô hình:

RMSE, MAE (rating)

Precision@K, Recall@K, NDCG (top-N)

Có thể viết evaluate_model(model, test_data)

utils.py

Hàm phụ: encode/decode user_id, chuyển đổi ma trận, logging, etc.

Để tái sử dụng tiện lợi ở các file khác

✅ Tách từng file giúp modular hóa, dễ debug, dễ mở rộng.

📁 app/ — GIAO DIỆN & API
api.py

Dùng Flask hoặc FastAPI để tạo endpoint:

GET /recommend?user_id=123 → trả về list phim gợi ý

Có thể thêm các endpoint phụ như:

/health

/similar_movies?movie_id=456

ui.py

Dùng Streamlit hoặc Gradio để tạo giao diện người dùng:

Người dùng chọn user → hiện danh sách phim đề xuất

Có thể hiện ảnh poster, rating, tên phim...

✅ Tách phần giao tiếp với người dùng giúp deploy dễ hơn, hoặc tích hợp vào hệ thống khác.

📁 tests/

Chứa unit tests cho các hàm trong src/

Dùng pytest hoặc unittest

Ví dụ:

Kiểm tra hàm recommend_by_cf() có trả ra đúng số lượng kết quả không

Hàm get_similar_movies() có lỗi khi input sai không

✅ Testing giúp bạn đảm bảo code chạy đúng khi refactor hoặc thêm tính năng.

📄 requirements.txt

Danh sách thư viện Python cần cài đặt:

pandas
numpy
scikit-learn
surprise
streamlit
flask

✅ Giúp người khác (hoặc chính bạn) cài đặt đúng môi trường.

📄 README.md

Mô tả project:

Mục tiêu

Cách chạy

Cấu trúc thư mục

Hướng dẫn cài đặt

Link demo nếu có

✅ Rất quan trọng nếu bạn chia sẻ code cho người khác hoặc làm portfolio cá nhân.

📄 main.py

File chạy chính của toàn bộ hệ thống

Có thể dùng để:

Load dữ liệu

Train mô hình

Chạy gợi ý cho 1 user cụ thể

Hoặc chạy API (from app.api import app)

if **name** == "**main**":
from src.collaborative import recommend_by_cf
print(recommend_by_cf(user_id=42))

✅ Giống như "cửa ngõ" của toàn bộ hệ thống.
