from fastapi import FastAPI, Request
from src.planner import Planner

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat")
async def dummy_chat(request: Request):
    request_JSON = await request.json()
    message = request_JSON["message"]
    if "history" in request_JSON:
        history = request_JSON["history"]
    _Planner = Planner()
    reply = await _Planner.run(question=message)
    if type(reply) == dict:
        return reply
    else:
        return {
            "answer": str(reply),
            }
