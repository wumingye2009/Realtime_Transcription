from pydantic import BaseModel


class SessionEvent(BaseModel):
    event_type: str
    message: str
