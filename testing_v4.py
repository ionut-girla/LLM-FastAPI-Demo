import requests
import json
import yaml
from app.models import ResponseModel


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
        "content": f"What is the meaning of life?"
      }
    ],
    
  })
)
print(response.text)

cleaned_json_str = response.text.strip()
parsed = json.loads(cleaned_json_str)

# Use the Pydantic model
response_obj = ResponseModel(**parsed)
answer = response_obj.extract_answer()

print(answer)