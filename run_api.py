import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Docs available at http://localhost:8000/docs")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
