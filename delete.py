from fastapi import FastAPI, File, UploadFile, Request,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import List
from pathlib import Path
import uuid
import os

#删除图片接口
app = FastAPI()

UPLOAD_DIR=""
@app.delete("/images/{image_name}")
async def delete_image(image_name: str):
    image_path = os.path.join(UPLOAD_DIR, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    os.remove(image_path)
    return {"message": "Image deleted"}