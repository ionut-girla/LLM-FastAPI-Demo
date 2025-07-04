from transformers import BitsAndBytesConfig
from app.helper import (summarize_weather, build_weather_prompt, get_weather_data)


lat = 44.4268
lon = 26.1025
location_name = "buc"
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
print(f"loading gpt2")
print(f"google/flan-t5-base")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

weather_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
weather_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


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

# Generate suggestion using the same model
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

print(suggestion)
json_result = {
    "location": location_name,
    "summary": {
        "today": today,
        "tomorrow": tomorrow
    },
    "suggestion": suggestion
}
print(json_result)
