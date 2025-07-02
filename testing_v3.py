from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime, timedelta
import statistics
import torch
import httpx
from helper import (reverse_geocode, summarize_weather, build_weather_prompt, get_weather_data)


lat = 44.4268
lon = 26.1025
location_name = "buc"
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

weather_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
weather_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


data = get_weather_data(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,uv_index,precipitation,wind_speed_10m"
        )
        # print(data)

current = data.get("current", {})
temp = current.get("temperature_2m")
humidity = current.get("relative_humidity_2m")
uv = current.get("uv_index")
rain = current.get("precipitation")
wind = current.get("wind_speed_10m")

# 🔧 Build natural language prompt
prompt = (
    f"You are a helpful assistant that gives daily advice based on the weather forecast. The weather forecast for today is:\n"
    f"- Temperature: {temp} degrees Celsius\n"
    f"- Humidity: {humidity}%\n"
    f"- UV Index: {uv}\n"
    f"- Precipitation: {rain} mm\n"
    f"- Wind Speed: {wind} km/h\n\n"
    f"Answer like a pirate"
    f"Based on this forecast, suggest what someone should wear, if they should wear sunscreen and what activities they could do (like walking, cycling, swimming)."
)


# 🔮 Generate suggestion using the same model
inputs = weather_tokenizer(prompt, return_tensors="pt").to(weather_model.device)
output = weather_model.generate(**inputs, max_new_tokens=128, pad_token_id=weather_tokenizer.eos_token_id, eos_token_id=weather_tokenizer.eos_token_id, no_repeat_ngram_size=3)
full_text = weather_tokenizer.decode(output[0], skip_special_tokens=True)
print(full_text)
# Extract only the generated suggestion (remove prompt part)
suggestion = full_text[len(prompt):].strip()
print('\n')
print(suggestion)
