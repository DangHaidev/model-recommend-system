from fastapi import FastAPI
from fastapi_utils.tasks import repeat_every
from fastapi import FastAPI, HTTPException
from fastapi import Query
import sys
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.content_based.cb_movie import (
    recommend_similar_movies,
)
from models.content_based.cb_user_profile import (
    recommend_movies as recommend_user_profile
)
from src.pipeline import run_pipeline

app = FastAPI(
    title="Movie Recommendation API",
    description="Content-based movie recommendation system",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    """
    Chạy 1 lần duy nhất khi app start
    """
    try:
        init_model()
        print("✅ Recommendation model initialized")
    except Exception as e:
        print("❌ Failed to initialize model:", e)

@app.on_event("startup")
@repeat_every(seconds=300, wait_first=True)
def batch_job() -> None:
    """
    Batch pipeline chạy nền, không block API
    """
    try:
        print("🚀 Running recommendation pipeline...")
        run_pipeline()
        print("✅ Pipeline finished")
    except Exception as e:
        print("❌ Pipeline error:", e)

@app.get("/health")
def health_check():
    return {"status": "ok"}
@app.get("/recommend/contentbased/{movie_id}")
def recommend_movies(movie_id: int,top_n: int = Query(10, ge=1)):
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
@app.get("/recommend/userprofile/{user_id}")
def recommend_movies2( user_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
    wm: float = 1.0,
    wu: float = 1.0):
    """
    Get top-N similar movies by user_id
    """
    try:
        df, total = recommend_user_profile( user_id=user_id,
        page=page,
        page_size=page_size,
        wm=wm,
        wu=wu)

        return {
        "user_id": user_id,
        "page": page,
        "page_size": page_size,
        "total_items": total,
        "total_pages": (total + page_size - 1) // page_size,
        "recommendations": df.to_dict(orient="records")
    }

    except IndexError:
        raise HTTPException(
            status_code=404,
            detail=f"Movie ID {user_id} a not found"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
