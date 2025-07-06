from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os
import json
from .helper import load_yaml_data
# Use your actual Mongo URI

MONGO_URI = load_yaml_data("config.yaml").get("mongo_uri")
client = AsyncIOMotorClient(MONGO_URI)
db = client["assistant_logs"]

# Log question and answer (for /ask)
async def log_ask(question: str, answer: str):
    await db.ask_logs.insert_one({
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    })

# Log weather suggestion only

async def log_weather(response_dict: dict, day: str):  # day can be "today" or "tomorrow"
    await db.weather_logs.insert_one({
        "day": day,
        "timestamp": datetime.utcnow(),
        "content": response_dict
        # "content": json.dumps(response_dict, ensure_ascii=False, indent=2)
    })
