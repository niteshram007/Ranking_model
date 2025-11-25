from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from src.server.websocket import websocket_endpoint, predictor
from src.ranking.model import RankingService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ranking_service = RankingService()


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.get('/predict')
async def predict(ticker: str):
    if not ticker:
        raise HTTPException(status_code=400, detail='ticker required')
    pred = await predictor.predict(ticker)
    return {'ticker': ticker, 'prediction': float(pred)}


class PredictRequest(BaseModel):
    # 2D array: list of rows, each row is list[float] of features (Close, MA10, ...)
    data: List[List[float]]
    ticker: Optional[str] = None


class TopPicksRequest(BaseModel):
    tickers: List[str]
    categories: Optional[List[str]] = None
    benchmark: Optional[str] = None
    top_k: int = 10


@app.post('/predict_data')
async def predict_data(req: PredictRequest):
    # validate input
    if not req.data or len(req.data) == 0:
        raise HTTPException(status_code=400, detail='data array required')
    try:
        pred = predictor.predict_from_array(req.data)
        return {'ticker': req.ticker, 'prediction': float(pred)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/top_picks')
async def top_picks(req: TopPicksRequest):
    if not req.tickers:
        return []
    if not ranking_service.is_ready():
        raise HTTPException(status_code=503, detail='ranking model not available')
    try:
        rows = ranking_service.top_picks(
            tickers=req.tickers,
            categories=req.categories,
            benchmark=req.benchmark,
            top_k=req.top_k,
        )
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.add_websocket_route("/ws", websocket_endpoint)