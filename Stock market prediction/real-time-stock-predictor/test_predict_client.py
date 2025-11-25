from fastapi.testclient import TestClient
from src.server.main import app

client = TestClient(app)
rows = [[100.0 + i * 0.1 for _ in range(6)] for i in range(60)]
resp = client.post('/predict_data', json={'ticker': 'AAPL', 'data': rows})
print('STATUS', resp.status_code)
print(resp.text)
