from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import json

from inference_engine import decode_greedy  # your model

app = FastAPI()

# Serve static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(req: Request):
    data = await req.json()
    msg = data.get("message", "")
    reply = decode_greedy(msg)
    return {"reply": reply}
