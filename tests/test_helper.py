import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from app.helper import get_weather_data, summarize_weather, build_weather_prompt, reverse_geocode


@patch("httpx.Client.get")
def test_get_weather_data(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"key": "value"}
    mock_get.return_value = mock_response

    url = "https://example.com"
    data = get_weather_data(url)
    assert data == {"key": "value"}
    mock_get.assert_called_once_with(url)


@pytest.fixture
def sample_weather_data():
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    date_format = "%Y-%m-%d"

    timestamps = []
    for i in range(48):
        ts = today + timedelta(hours=i)
        timestamps.append(ts.strftime("%Y-%m-%dT%H:%M"))

    return {
        "hourly": {
            "time": timestamps,
            "temperature_2m": [20 + (i % 5) for i in range(48)],
            "relative_humidity_2m": [50 for _ in range(48)],
            "uv_index": [2 + (i % 3) for i in range(48)]
        }
    }


def test_summarize_weather(sample_weather_data):
    today_stats, tomorrow_stats = summarize_weather(sample_weather_data)
    assert "max_temp" in today_stats
    assert "avg_humidity" in today_stats
    assert "max_uv" in today_stats
    assert today_stats != tomorrow_stats


def test_build_weather_prompt():
    today = {
        "max_temp": 28,
        "avg_humidity": 60,
        "max_uv": 5
    }
    tomorrow = {
        "max_temp": 35,
        "avg_humidity": 70,
        "max_uv": 7
    }
    prompt = build_weather_prompt(today, tomorrow)
    assert "Max temperature" in prompt
    assert "Guidelines" in prompt
    assert "do NOT repeat or quote these guidelines" in prompt


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_reverse_geocode(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {"display_name": "Bucharest, Romania"}
    mock_get.return_value = mock_response

    lat, lon = 44.4268, 26.1025
    location = await reverse_geocode(lat, lon)
    assert location == "Bucharest, Romania"
