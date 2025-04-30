import logging
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

task_id = str(uuid.uuid4())
upload_dir = Path("static/uploads") / task_id
upload_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# 允许的图片文件后缀
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.tiff', '.tif'}


def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/images/{filename}")
async def get_image(filename: str):
    if not is_allowed_file(filename):
        logger.error(f"不允许的文件类型: {filename}")
        return {"error": "不允许的文件类型"}
    file_location = upload_dir / filename
    try:
        if file_location.exists():
            logger.info(f"成功读取图片: {file_location}")
            return FileResponse(file_location)
        else:
            logger.error(f"未找到图片: {file_location}")
            return {"error": "文件未找到"}
    except Exception as e:
        logger.error(f"读取文件时出错: {e}")
        return {"error": "读取文件时出错"}





