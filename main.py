# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Buat app FastAPI
app = FastAPI(title="MakeOver Backend")

# Tambahkan CORS supaya frontend bisa request
origins = [
    "http://localhost:3000",               # frontend lokal
    "http://localhost:5173",  
    "https://makeover-frontend.vercel.app" # frontend online nanti
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

# Contoh endpoint API
@app.get("/api/hello")
def say_hello():
    return {"message": "Hello from FastAPI backend!"}
