from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import json
from fastapi.middleware.cors import CORSMiddleware


from inference_engine import decode_greedy  # your model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
