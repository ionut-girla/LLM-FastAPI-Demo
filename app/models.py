from pydantic import BaseModel
from typing import List, Optional

class Question(BaseModel):
    question: str

class Message(BaseModel):
    role: str
    content: str

    @property
    def cleaned_content(self) -> str:
        # Extract content after last ◁/think▷ tag
        if "◁/think▷" in self.content:
            return self.content.split("◁/think▷")[-1].strip()
        return self.content.strip()

class Choice(BaseModel):
    message: Message

class ResponseModel(BaseModel):
    choices: List[Choice]

    @property
    def final_answer(self) -> Optional[str]:
        if self.choices:
            return self.choices[0].message.cleaned_content
        return None
    # def extract_answer(self) -> str:
    #     # Safely get the content of the first message
    #     return self.choices[0].message.content if self.choices else ""