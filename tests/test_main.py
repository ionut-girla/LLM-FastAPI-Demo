import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.app import app  # replace with the actual name of your main file, e.g., `from main import app`

client = TestClient(app)


@pytest.mark.asyncio
@patch("app.httpx.AsyncClient.get")
@patch("app.tokenizer")
@patch("app.model")
async def test_weather_suggestion(mock_model, mock_tokenizer, mock_get):
    # Mock the weather API response
    mock_get.return_value.__aenter__.return_value.json.return_value = {
        "current": {
            "temperature_2m": 30,
            "relative_humidity_2m": 65,
            "uv_index": 7,
            "precipitation": 0.2,
            "wind_speed_10m": 15
        }
    }

    # Mock tokenizer behavior
    mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
    mock_tokenizer.return_tensors = "pt"
    mock_tokenizer.return_value.to.return_value = {"input_ids": [[1, 2, 3]]}
    mock_tokenizer.decode.return_value = (
        "The weather today in Bucharest is:\n"
        "- Temperature: 30°C\n"
        "- Humidity: 65%\n"
        "- UV Index: 7\n"
        "- Precipitation: 0.2 mm\n"
        "- Wind Speed: 15 km/h\n\n"
        "Based on this data, give a short, friendly suggestion on how someone should prepare for the day.\n"
        "It's warm and sunny today. Wear light clothing, stay hydrated, and enjoy the outdoors!"
    )

    # Mock model output
    mock_model.device = "cpu"
    mock_model.generate.return_value = [[1, 2, 3, 4]]

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/weather-suggestion")

    assert response.status_code == 200
    data = response.json()

    assert "suggestion" in data
    assert "temperature" in data
    assert data["location"] == "Bucharest"
