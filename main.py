import numpy as np
import pandas as pd
from src.data_loader import load_ratings, load_movies
from src.collaborative import CF, get_dataframe_ratings_base


def main():
    print("🚀 Bắt đầu mô hình Collaborative Filtering\n")

    # ----------------------------------------------------------
    # 1️⃣ LOAD DỮ LIỆU
    # ----------------------------------------------------------
    print("📂 Đang load dữ liệu...")
    Y_data = get_dataframe_ratings_base()
    movies = load_movies()

    print(f"✅ Kích thước dữ liệu ratings: {Y_data.shape}")
    print("5 dòng đầu tiên:")
    print(pd.DataFrame(Y_data, columns=["UserID", "MovieID", "Rating"]).head())

    # ----------------------------------------------------------
    # 2️⃣ KHỞI TẠO MÔ HÌNH CF
    # ----------------------------------------------------------
    k = 20
    uuCF = 1  # 1 = user-user CF, 0 = item-item CF

    model = CF(data_matrix=Y_data, k=k, uuCF=uuCF)

    # ----------------------------------------------------------
    # 3️⃣ CHUẨN HÓA MA TRẬN
    # ----------------------------------------------------------
    print("\n🔧 Chuẩn hóa ma trận...")
    model.normalize_matrix()

    print("===== 📊 MA TRẬN CHUẨN HÓA (Ybar - r - mean_user) =====")
    df_Ybar = pd.DataFrame(
        model.Ybar.toarray()[:10, :10],
        columns=[f"User_{i+1}" for i in range(10)],
        index=[f"Item_{i+1}" for i in range(10)]
    )
    print(df_Ybar.round(3))

    # ----------------------------------------------------------
    # 4️⃣ TÍNH MA TRẬN TƯƠNG ĐỒNG
    # ----------------------------------------------------------
    print("\n🔗 Tính toán ma trận tương đồng (similarity)...")
    model.similarity()

    print("===== 🔗 MA TRẬN TƯƠNG ĐỒNG (S) =====")
    n_show = min(10, model.S.shape[0])
    df_sim = pd.DataFrame(
        model.S[:n_show, :n_show],
        columns=[f"User_{i+1}" for i in range(n_show)],
        index=[f"User_{i+1}" for i in range(n_show)]
    )
    print(df_sim.round(3))

    # ----------------------------------------------------------
    # 5️⃣ GỢI Ý PHIM CHO USER
    # ----------------------------------------------------------
    user_id = 10  # User thứ 11 thực tế (vì index bắt đầu từ 0)
    top_n = 10
    print(f"\n🎬 Gợi ý top {top_n} phim cho User_{user_id + 1}:")
    recommendations = model.recommend_top(user_id, top_x=top_n)

    print("--------------------------------------------------")
    for idx, rec in enumerate(recommendations, start=1):
        movie_id = rec['id']
        pred_rating = rec['pred_rating']

        # Lấy tên phim từ DataFrame movies
        movie_name = movies.loc[movies['ID'] == movie_id + 1, 'Title'].values
        movie_name = movie_name[0] if len(movie_name) > 0 else "Unknown"

        print(f"{idx:02d}. {movie_name} → dự đoán: {pred_rating:.2f}")

    print("--------------------------------------------------")
    print("🏁 Hoàn tất!")


if __name__ == "__main__":
    main()
