from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MakeOver Backend")

origins = [
    "http://localhost:3000",
    "http://localhost:5173", 
    "https://make-over-one.vercel.app",  # Frontend Anda
    "https://your-actual-project.up.railway.app"  # Domain Railway baru
]

# ALLOW SEMUA ORIGINS DULU UNTUK TESTING
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Izinkan semua domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

@app.get("/api/hello")
def say_hello():
    return {"message": "Hello from FastAPI backend!"}

# Tambahkan bagian ini di akhir file
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)