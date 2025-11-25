from fastapi import WebSocket
import asyncio
from fastapi import WebSocket
import asyncio
import json
from pathlib import Path
import joblib
import numpy as np
from typing import Sequence, Optional


class StockPredictor:
    def __init__(self):
        # Attempt to load a scaler and model from src/models directory
        self.models_dir = Path(__file__).resolve().parents[1] / 'models'
        # fallback to project-root models/ as well
        if not self.models_dir.exists():
            self.models_dir = Path(__file__).resolve().parents[2] / 'models'

        self.scaler: Optional[object] = None
        self.model: Optional[object] = None
        try:
            scaler_path = self.models_dir / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
        except Exception:
            self.scaler = None

        # Try to find any .h5 keras model
        try:
            h5 = list(self.models_dir.glob('*.h5'))
            if h5:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(h5[0])
        except Exception:
            self.model = None

    async def predict(self, ticker: str) -> float:
        """Fallback prediction by running a zero-input through the model (if available)."""
        await asyncio.sleep(0.1)
        if self.model is not None:
            try:
                input_shape = self.model.input_shape
                seq_len = input_shape[1] or 60
                n_features = input_shape[2] or (self.scaler.n_features_in_ if self.scaler is not None else 6)
                inp = np.zeros((1, seq_len, n_features), dtype=np.float32)
                pred = self.model.predict(inp, verbose=0)
                if self.scaler is not None:
                    dummy = np.zeros((1, self.scaler.n_features_in_))
                    dummy[0, 0] = float(pred.ravel()[0])
                    inv = self.scaler.inverse_transform(dummy)
                    return float(inv[0, 0])
                return float(pred.ravel()[0])
            except Exception:
                return 100.0
        return 100.0

    def predict_from_array(self, arr: Sequence[Sequence[float]]) -> float:
        """Accept a 2D array-like of shape (seq_len, n_features), scale, predict, and inverse-transform the output.

        Returns a float prediction.
        """
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError('data must be 2D array: (seq_len, n_features)')

        # If scaler exists, transform features
        a_scaled = a
        if self.scaler is not None:
            try:
                a_scaled = self.scaler.transform(a)
            except Exception:
                a_scaled = a

        # Predict
        if self.model is not None:
            try:
                inp = np.expand_dims(a_scaled, axis=0)
                pred = self.model.predict(inp, verbose=0)
                val = float(pred.ravel()[0])
                if self.scaler is not None:
                    dummy = np.zeros((1, self.scaler.n_features_in_))
                    dummy[0, 0] = val
                    inv = self.scaler.inverse_transform(dummy)
                    return float(inv[0, 0])
                return val
            except Exception:
                return 100.0
        return 100.0


predictor = StockPredictor()


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            ticker = request.get("ticker")
            prediction = await predictor.predict(ticker)
            await websocket.send_text(json.dumps({"ticker": ticker, "prediction": prediction}))
    except Exception:
        await websocket.close()