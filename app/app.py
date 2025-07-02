# filename: app.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from datetime import datetime, timedelta
import statistics
import torch
import httpx
from .helper import (
    reverse_geocode,
    summarize_weather,
    build_weather_prompt,
    get_weather_data,
)
app = FastAPI(title="Code QA Assistant")

@app.on_event("startup")
def load_model_once():
    global weather_tokenizer, code_tokenizer, code_model, weather_model

    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    # print(f"loading gpt2")
    # weather_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # weather_tokenizer.pad_token = weather_tokenizer.eos_token
    # weather_model = AutoModelForCausalLM.from_pretrained("gpt2")
    # weather_model.eval()

    print(f"google/flan-t5-base")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    weather_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    weather_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    print("loading starcoder")
    code_tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
    code_tokenizer.pad_token = code_tokenizer.eos_token
    code_model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b", quantization_config=bnb_config)


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(q: Question):
    try:
        prompt = f"### Question:\n{q.question}\n\n### Answer:\n"
        # to use 4bit use `load_in_4bit=True` instead
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        inputs = code_tokenizer(prompt, return_tensors="pt").to(code_model.device)

        # Fix pad_token_id and attention_mask
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        output = code_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy decoding for determinism
            pad_token_id=code_tokenizer.eos_token_id,
            eos_token_id=code_tokenizer.eos_token_id
        )

        answer = code_tokenizer.decode(output[0], skip_special_tokens=True)
        final_answer = answer.split("### Answer:")[-1].strip()
        return {"answer": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather-suggestion-today")
async def weather_suggestion():
    try:
        lat = 44.4268
        lon = 26.1025
        location_name = await reverse_geocode(lat, lon)

        # Fetch weather data
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
        output = weather_model.generate(**inputs, max_new_tokens=128, pad_token_id=weather_tokenizer.eos_token_id,
                                        eos_token_id=weather_tokenizer.eos_token_id, no_repeat_ngram_size=3)
        full_text = weather_tokenizer.decode(output[0], skip_special_tokens=True)

        # # Extract only the generated suggestion (remove prompt part)
        # suggestion = full_text[len(prompt):].strip()

        return {
            "location": location_name,
            "temperature": temp,
            "humidity": humidity,
            "uv_index": uv,
            "precipitation": rain,
            "wind_speed": wind,
            "suggestion": full_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Weather model error: {str(e)}")


@app.get("/weather-suggestion-tomorrow")
async def weather_suggestion():
    try:
        lat = 44.4268
        lon = 26.1025
        location_name = await reverse_geocode(lat, lon)

        # Fetch weather data
        weather_data = get_weather_data(
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            "hourly=temperature_2m,relative_humidity_2m,uv_index&"
            "timezone=auto"
        )
        # print(weather_data)
        # 2. Summarize data
        today, tomorrow = summarize_weather(weather_data)
        prompt = build_weather_prompt(today, tomorrow)

        # 🔮 Generate suggestion using the same model
        inputs = weather_tokenizer(prompt, return_tensors="pt").to(weather_model.device)
        outputs = weather_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            pad_token_id=weather_tokenizer.eos_token_id,
            eos_token_id=weather_tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )

        suggestion = weather_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return {
            "location": location_name,
            "summary": {
                "tomorrow": tomorrow
            },
            "suggestion": f"Considering the weather forcast for tomorrow, I would suggest: {suggestion}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))