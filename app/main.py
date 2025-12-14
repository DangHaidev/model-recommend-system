from fastapi import FastAPI, HTTPException
import sys
from pathlib import Path

# ===============================
# Add project root to PYTHONPATH
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ===============================
# Import recommendation function
# ===============================
from models.content_based.cb_movie import (
    recommend_similar_movies,
    init_model  # nếu bạn chưa có thì có thể bỏ dòng này
)

# ===============================
# Create FastAPI app
# ===============================
app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendation system",
    version="1.0.0"
)

# ===============================
# Startup event
# ===============================
@app.on_event("startup")
def startup_event():
    """
    Load model / compute similarity matrix once
    """
    try:
        init_model()
        print("✅ Recommendation model initialized")
    except Exception as e:
        print("❌ Failed to initialize model:", e)

# ===============================
# Health check
# ===============================
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ===============================
# Recommendation endpoint
# ===============================
@app.get("/recommend/{movie_id}")
def recommend_movies(movie_id: int, top_n: int = 10):
    """
    Get top-N similar movies by movie_id
    """
    try:
        df = recommend_similar_movies(movie_id, top_n)

        return {
            "movie_id": movie_id,
            "top_n": top_n,
            "recommendations": df.to_dict(orient="records")
        }

    except IndexError:
        raise HTTPException(
            status_code=404,
            detail=f"Movie ID {movie_id} not found"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
