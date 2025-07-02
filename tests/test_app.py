import pytest
from unittest.mock import patch
import json


@pytest.mark.asyncio
@patch("app.helper.reverse_geocode", return_value="Bucharest, Romania")
@patch("app.helper.get_weather_data")
@patch("app.helper.summarize_weather")
@patch("app.helper.build_weather_prompt")
def test_weather_suggestion_tomorrow(
    mock_prompt,
    mock_summary,
    mock_weather_data,
    mock_reverse_geocode,
    client
):
    mock_weather_data.return_value = {
        "hourly": {
            "time": ["2025-07-02T00:00", "2025-07-02T01:00"],
            "temperature_2m": [25, 28],
            "relative_humidity_2m": [30, 35],
            "uv_index": [3, 6]
        }
    }
    mock_summary.return_value = (
        {"max_temp": 28, "avg_humidity": 32, "max_uv": 6},
        {"max_temp": 32, "avg_humidity": 40, "max_uv": 8}
    )
    mock_prompt.return_value = "Sample prompt for tomorrow"

    response = client.get("/weather-suggestion-tomorrow")
    assert response.status_code == 200
    assert "suggestion" in response.json()


@pytest.mark.asyncio
@patch("app.helper.reverse_geocode", return_value="Bucharest, Romania")
@patch("app.helper.get_weather_data")
def test_weather_suggestion_today(mock_weather_data, mock_reverse_geocode, client):
    mock_weather_data.return_value = {
        "current": {
            "temperature_2m": 31,
            "relative_humidity_2m": 40,
            "uv_index": 7,
            "precipitation": 0,
            "wind_speed_10m": 15
        }
    }

    response = client.get("/weather-suggestion-today")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["temperature"] == 31
    assert "suggestion" in json_data


def test_ask_question(client):
    response = client.post("/ask", json={"question": "What is a Python dict?"})
    assert response.status_code == 200
    assert "answer" in response.json()
