from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from typing import List
from pathlib import Path
import uuid

app = FastAPI()

# 挂载静态目录：将 /static 映射到本地的
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置 Jinja2 模板目录
#templates = Jinja2Templates(directory="templates")

'''前端
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
'''

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    # 为本次任务生成唯一 UUID
    task_id = str(uuid.uuid4()) #======================
    # 本次上传文件夹路径
    upload_dir = Path("static/uploads") / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 允许的文件后缀列表
    allowed_ext = (".tif", ".tiff", ".png", ".jpeg", ".jpg")
    urls = []
    for file in files:
        filename = file.filename
        if not filename.lower().endswith(allowed_ext):
            continue
        file_path = upload_dir / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        urls.append(f"/static/uploads/{task_id}/{filename}")
    # 返回任务 ID 与文件 URL 列表
    return {"task_id": task_id, "urls": urls}
