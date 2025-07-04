from datetime import datetime, timedelta
import statistics

import yaml
from app.models import Question, ResponseModel
import httpx
import requests
import json
import re

def get_weather_data(weather_url):
    with httpx.Client() as client:
        res = client.get(weather_url)
        return res.json()


# def build_weather_prompt(today, tomorrow):
#     return f"""You are a helpful assistant that gives daily advice based on the weather forecast.
#
# Tomorrow:
# - Max temp: {tomorrow['max_temp']} degrees Celsius
# - Avg humidity: {tomorrow['avg_humidity']}%
# - Max UV index: {tomorrow['max_uv']}
#
# Based on this forecast, suggest what someone should wear, if they should wear sunscreen and what activities they could do (like walking, cycling, swimming)."""


def build_weather_prompt(today, tomorrow) -> Question:
    return Question(question=f"""
        You are a helpful assistant that gives daily advice based on the weather forecast for Bucharest.

        Tomorrow's forecast:
        - Max temperature: {tomorrow['max_temp']} degrees Celsius
        - Average humidity: {tomorrow['avg_humidity']}%
        - Max UV index: {tomorrow['max_uv']}

        Guidelines you must *use* to form your advice:
        - If temperature > 30°C: Recommend light, breathable clothes and staying hydrated.
        - If UV index ≥ 6: Recommend sunscreen and a hat.
        - If humidity > 20%: Recommend carrying an umbrella.
        - If wind speed > 20 km/h: Warn about strong winds.

        **Important:** Do NOT repeat or quote these guidelines. Instead, based on them and the forecast above, write a friendly suggestion on what to wear and activities to do or avoid tomorrow for someone else. Keep your response short and natural (maximum 3 sentences).
        """)

def summarize_weather(data):
    hourly = data["hourly"]
    timestamps = hourly["time"]
    temps = hourly["temperature_2m"]
    humidity = hourly["relative_humidity_2m"]
    uv = hourly["uv_index"]

    # Get today's and tomorrow's date in YYYY-MM-DD format
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    def extract_day_stats(day_str):
        day_temps = []
        day_humidity = []
        day_uv = []

        for i, ts in enumerate(timestamps):
            if ts.startswith(day_str):
                day_temps.append(temps[i])
                day_humidity.append(humidity[i])
                day_uv.append(uv[i])

        if not day_temps:
            return None  # Handle empty case safely

        return {
            "min_temp": min(day_temps),
            "max_temp": max(day_temps),
            "avg_humidity": round(statistics.mean(day_humidity), 1),
            "max_uv": max(day_uv)
        }

    today_stats = extract_day_stats(today_str)
    tomorrow_stats = extract_day_stats(tomorrow_str)

    return today_stats, tomorrow_stats


async def reverse_geocode(lat: float, lon: float) -> str:
    """Get location name from latitude/longitude using Nominatim."""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    headers = {"User-Agent": "weather-suggestion-app"}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        data = response.json()
        return data.get("display_name", "Unknown location")

def load_yaml_data(filename):
    """
    Load Yaml data from a file
    """
    try:
        with open(filename, encoding="utf8") as conf:
            data = yaml.safe_load(conf)
            return data
    except OSError:
        print("err")
        return {}

def limit_to_3_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(sentences[:3])

def get_response_from_api(q: Question):
    api_key = load_yaml_data("config.yaml").get("api_key")
    response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "moonshotai/kimi-dev-72b:free",
                "messages": [
                {
                    "role": "user",
                    "content": f"Respond in at most 3 sentences to the following query {q.question}"
                }
                ],
                
            })
            )

    cleaned_json_str = response.text.strip()
    parsed = json.loads(cleaned_json_str)

    # Use the Pydantic model
    response_obj = ResponseModel(**parsed)
    answer = response_obj.final_answer

    return answer