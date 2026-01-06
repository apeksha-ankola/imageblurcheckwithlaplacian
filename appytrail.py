from fastapi import FastAPI, Request, HTTPException
import cv2
import numpy as np
import uvicorn

app = FastAPI(title="Blur Detection API (Binary)")

def is_image_blurry(image_bytes: bytes, threshold: float = 200.0):
    # Convert bytes → numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode as image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, 0.0

    # Laplacian variance calculation
    lap = cv2.Laplacian(img, cv2.CV_64F)
    variance = float(lap.var())

    # Blurry if variance < threshold
    blurry = bool(variance < threshold)
    return blurry, variance

@app.post("/check-blur")
async def check_blur(request: Request):
    """
    Accepts raw binary image data in the request body.
    """
    try:
        # This is the FastAPI equivalent of Flask's request.data
        image_bytes = await request.body()
        
        if not image_bytes:
            raise HTTPException(status_code=400, detail="No image data received")

        blurry, score = is_image_blurry(image_bytes)
        
        if blurry is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        return {
            "isBlurry": blurry,
            "score": round(score, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Note: Using port 8000 as per your previous setup
    uvicorn.run(app, host="localhost", port=8000)