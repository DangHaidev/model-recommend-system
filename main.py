import pandas as pd
from src.collaborative import CF, get_dataframe_ratings_base

def main():
    # 1️⃣ Đọc dữ liệu ratings (user, item, rating)
    print("🔹 Đang tải dữ liệu ratings...")
    Y_data = get_dataframe_ratings_base()
    print(f"Tổng số dòng dữ liệu: {len(Y_data)}")

    # 2️⃣ Khởi tạo mô hình CF
    print("🔹 Khởi tạo mô hình Collaborative Filtering...")
    k = 30   # số lượng user giống nhất (neighbors)
    rs = CF(Y_data, k=k, uuCF=1)  # uuCF=1: user-user CF, uuCF=0: item-item CF

    # 3️⃣ Huấn luyện mô hình
    print("🔹 Đang chuẩn hóa và tính độ tương đồng...")
    rs.normalize_matrix()
    rs.similarity()
    print("✅ Huấn luyện xong!")

    # 4️⃣ Gợi ý cho một user cụ thể (ví dụ user_id = 10)
    user_id = 10
    top_n = 10
    print(f"🔹 Gợi ý top {top_n} phim cho user {user_id} ...")

    recommendations = rs.recommend_top(u=user_id, top_x=top_n)

    # 5️⃣ In kết quả
    print("\n🎬 Danh sách phim được gợi ý:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx:02d}. MovieID: {rec['id']}, Predicted Rating: {rec['pred_rating']:.4f}")



if __name__ == "__main__":
    main()
