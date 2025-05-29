from fastapi import FastAPI, File, UploadFile, HTTPException, Query,Response, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import shutil
import os
import io
import pandas as pd
import uvicorn
import cv2
import tempfile

# micro_sam
import os
from glob import glob
from typing import Optional, Union, Tuple
from PIL import Image
from typing import List
from skimage import measure
from typing import List, Dict
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components
from scipy.stats import scoreatpercentile

import torch

from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from pydantic import BaseModel
import uuid
import json

def run_automatic_instance_segmentation(
    image: np.ndarray,
    ndim: int,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    model_type: str = "vit_b_lm",
    device: Optional[Union[str, torch.device]] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
):
    """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

    NOTE: AIS is supported only for `µsam` models.

    Args:
        image: The input image.
        ndim: The number of dimensions for the input data.
        checkpoint_path: The path to stored checkpoints.
        model_type: The choice of the `µsam` model.
        device: The device to run the model inference.
        tile_shape: The tile shape for tiling-based segmentation.
        halo: The overlap shape on each side per tile for stitching the segmented tiles.

    Returns:
        The instance segmentation.
    """
    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
        device=device,  # the device to run the model inference.
        amg=False,  # set the automatic segmentation mode to AIS.
        is_tiled=(tile_shape is not None),  # whether to run automatic segmentation with tiling.
    )

    # Step 2: Get the instance segmentation for the given image.
    prediction = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,  # the filepath to image or the input array for automatic segmentation.
        ndim=ndim,  # the number of input dimensions.
        tile_shape=tile_shape,  # the tile shape for tiling-based prediction.
        halo=halo,  # the overlap shape for tiling-based prediction.
    )

    return prediction

def calculate_region_properties(labels):
    """
    计算每个区域的指定属性。
    """
    regions = measure.regionprops(labels)
    properties = []
    for region in regions:
        # 提取基本属性
        area = region.area
        perimeter = region.perimeter
        bbox = region.bbox
        major_axis_length = region.major_axis_length
        minor_axis_length = region.minor_axis_length
        eccentricity = region.eccentricity

        # 计算派生属性
        diameter = np.sqrt(area / np.pi) * 2  # 粒径（等效圆直径）
        aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else 0  # 长宽比
        sphericity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0  # 球形度
        shape_factor = area / (major_axis_length ** 2) if major_axis_length > 0 else 0  # 形状因子 A/R
        smoothness = 1 - (eccentricity)  # 平滑度（1 - 偏心率）

        # 存储属性
        props = {
            "area": area,
            "perimeter": perimeter,
            "diameter": diameter,
            "major_axis_length": major_axis_length,
            "minor_axis_length": minor_axis_length,
            "aspect_ratio": aspect_ratio,
            "sphericity": sphericity,
            "shape_factor": shape_factor,
            "smoothness": smoothness,
        }
        properties.append(props)
    
    return properties


def calculate_statistics(properties):
    """
    计算统计值：均值、标准差、最大值、最小值、P0、P10、P50、P90、P100。
    同时计算区域数量。
    """
    all_stats = {"region_count": len(properties)}  # 区域数量
    
    for key in properties[0].keys():  # 遍历属性名称
        values = [prop[key] for prop in properties]
        stats = {
            "mean": np.mean(values),
            "std": np.std(values),
            "max": np.max(values),
            "min": np.min(values),
            "P0": scoreatpercentile(values, 0),
            "P10": scoreatpercentile(values, 10),
            "P50": scoreatpercentile(values, 50),
            "P90": scoreatpercentile(values, 90),
            "P100": scoreatpercentile(values, 100),
        }
        all_stats[key] = stats
    
    return all_stats

# custom-docs-ui-assets
app = FastAPI(docs_url=None, redoc_url=None)

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://unpkg.com/redoc@2/bundles/redoc.standalone.js",
    )



# 创建一个目录用于保存上传的图片
UPLOAD_DIR = "uploads"
MODEL_DIR = "models/vit_b_lm"
SEGMENT_DIR = "segmentations"
model_choice = "vit_b_lm"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEGMENT_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "欢迎使用图片上传服务"}

# 任务存储的根目录路径
TASKS_ROOT_DIR_PATH = "tasks"

class Task(BaseModel):
    id: str
    name: str = "Untitled Task"
    description: str = ""

def ensure_task_dir_exists(task_id: str):
    task_dir_path = os.path.join(TASKS_ROOT_DIR_PATH, task_id)
    if not os.path.exists(task_dir_path):
        os.makedirs(task_dir_path)
    return task_dir_path

def load_task(task_id: str):
    task_file_path = os.path.join(TASKS_ROOT_DIR_PATH, task_id, "task.json")
    if not os.path.exists(task_file_path):
        return None
    with open(task_file_path, 'r') as file:
        try:
            task_data = json.load(file)
            task_data["id"] = task_id  # 确保 id 来自目录名
            return Task(**task_data)
        except json.JSONDecodeError:
            print(f"Invalid JSON in {task_file_path}")
            return None
    
        

def save_task(task: Task):
    ensure_task_dir_exists(task.id)
    task_file_path = os.path.join(TASKS_ROOT_DIR_PATH, task.id, "task.json")
    with open(task_file_path, 'w') as file:
        json.dump(task.model_dump(), file)  # 将 Task 模型转换为字典并保存


@app.post("/tasks/", response_model=dict)
async def create_task(task_json: str = Form(""),  # 设置默认值为空字符串
                      file: UploadFile = File(...)):
    # 如果 task_json 为空，则使用默认任务信息
    if not task_json.strip():
        task_dict = {"id":"talk some shit", "name": "Untitled Task", "description": ""}
    else:
        try:
            task_dict = json.loads(task_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for task")
    
    # 检查文件是否为图片（可选）
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "文件不是图片"})

    # 创建一个新的 Task 实例，并分配新的 task_id
    task = Task(**task_dict)
    task_id = str(uuid.uuid4())
    new_task = Task(id=task_id, name=task.name, description=task.description)
    save_task(new_task)

    task_dir = ensure_task_dir_exists(task_id)
    image_dir = os.path.join(task_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, file.filename)

     # 保存图片
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        prediction = run_automatic_instance_segmentation(image_path, ndim=2, model_type=model_choice)
        regionprops = calculate_region_properties(prediction)
        all_stats = calculate_statistics(regionprops)

        # 保存 numpy 文件
        np.save(os.path.join(task_dir, 'prediction.npy'), prediction)
        np.save(os.path.join(task_dir, 'regionprops.npy'), regionprops)
        np.save(os.path.join(task_dir, 'all_stats.npy'), all_stats)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像处理失败: {e}")

    return {"task_id": task_id, "message": "图片上传成功"}

@app.get("/tasks/{task_id}", response_model=Task)
def read_task(task_id: str):
    task_data = load_task(task_id)
    if task_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_data

@app.put("/tasks/{task_id}")
def update_task(task_id: str, task_update: Task):
    existing_task = load_task(task_id)
    if existing_task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    # 只更新提供了值的字段
    updated_data = existing_task.model_dump()  # 获取现有数据的字典形式
    update_data = task_update.model_dump(exclude_unset=True)  # 获取需要更新的数据，排除未设置的字段

    # 更新现有数据
    updated_data.update(update_data)

    # 确保 id 不被更改
    updated_data['id'] = task_id

    # 创建并保存更新后的任务
    updated_task = Task(**updated_data)
    save_task(updated_task)

    return {"message": "Task updated"}

@app.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    task_dir_path = os.path.join(TASKS_ROOT_DIR_PATH, task_id)
    if not os.path.exists(task_dir_path):
        raise HTTPException(status_code=404, detail="Task not found")
    import shutil
    shutil.rmtree(task_dir_path)
    return {"message": "Task deleted"}

@app.get("/tasks/", response_model=List[Task])
def list_all_task_attributes():
    if not os.path.exists(TASKS_ROOT_DIR_PATH):
        return []

    all_tasks = []

    for task_id in os.listdir(TASKS_ROOT_DIR_PATH):
        task_dir_path = os.path.join(TASKS_ROOT_DIR_PATH, task_id)

        # 只处理目录
        if not os.path.isdir(task_dir_path):
            continue

        task = load_task(task_id)
        if task is not None:
            all_tasks.append(task)

    return all_tasks

@app.get("/tasks/{task_id}/image/file")
async def get_image(task_id: str):
    """
    根据 task_id 自动查找并返回任务目录下唯一的图片。
    """
    task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id, "images")
    
    # 检查目录是否存在
    if not os.path.isdir(task_dir):
        raise HTTPException(status_code=404, detail="Task directory not found")

    # 获取目录中的所有文件
    images = [f for f in os.listdir(task_dir) if os.path.isfile(os.path.join(task_dir, f))]
    
    # 如果没有找到任何图片或找到多于一张图片，则抛出异常
    if len(images) != 1:
        raise HTTPException(status_code=400, detail="Expected exactly one image in the task directory")
    
    image_path = os.path.join(task_dir, images[0])

    # 读取原图
    original_image = cv2.imread(image_path)

    # 获取原始图像的扩展名
    suffix = Path(image_path).suffix

    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    cv2.imwrite(temp_file.name, original_image)
    
    return FileResponse(temp_file.name, filename=images[0])

# 返回原图叠加分割图层的图像
@app.get("/tasks/{task_id}/image/overlay")
async def get_overlay_image(task_id: str):
    try:
        # 获取任务目录
        task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id)

        # 获取原图路径
        image_dir = os.path.join(task_dir, "images")
        images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        if len(images) != 1:
            raise HTTPException(status_code=400, detail="Expected exactly one image in the task directory")
        image_path = os.path.join(image_dir, images[0])

        # 读取原图
        original_image = cv2.imread(image_path)

        # 读取分割结果
        prediction = np.load(os.path.join(task_dir, 'prediction.npy'))

        # 创建彩色分割图像
        colored_mask = np.zeros_like(original_image)
        unique_labels = np.unique(prediction)
        for label in unique_labels:
            if label == 0:  # 背景
                continue
            mask = (prediction == label).astype(np.uint8)
            color = np.random.randint(0, 256, 3).astype(np.uint8)
            colored_mask[mask == 1] = color

        # 叠加图像
        alpha = 0.5
        overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)

        # 保存叠加后的图像到临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_file.name, overlay)

        return FileResponse(temp_file.name, filename="overlay.png")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating overlay image: {str(e)}")

# 新增接口：切换分割图层的显隐

@app.get("/tasks/{task_id}/image/toggle_overlay")
async def toggle_overlay(task_id: str, show_overlay: bool = Query(True, description="是否显示分割图层")):
    try:
        # 获取任务目录
        task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id)

        # 获取原图路径
        image_dir = os.path.join(task_dir, "images")
        images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        if len(images) != 1:
            raise HTTPException(status_code=400, detail="Expected exactly one image in the task directory")
        image_path = os.path.join(image_dir, images[0])

        # 读取原图
        original_image = cv2.imread(image_path)

        if show_overlay:
            # 读取分割结果
            prediction = np.load(os.path.join(task_dir, 'prediction.npy'))

            # 创建彩色分割图像
            colored_mask = np.zeros_like(original_image)
            unique_labels = np.unique(prediction)
            for label in unique_labels:
                if label == 0:  # 背景
                    continue
                mask = (prediction == label).astype(np.uint8)
                color = np.random.randint(0, 256, 3).astype(np.uint8)
                colored_mask[mask == 1] = color

            # 叠加图像
            alpha = 0.5
            overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)
        else:
            overlay = original_image

        # 保存叠加后的图像到临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_file.name, overlay)

        return FileResponse(temp_file.name, filename="overlay.png")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating overlay image: {str(e)}")

# 新增接口：高亮点击的分割图层
@app.get("/tasks/{task_id}/image/highlight")
async def highlight_clicked_segment(task_id: str, x: int, y: int):
    try:
        # 获取任务目录
        task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id)

        # 获取原图路径
        image_dir = os.path.join(task_dir, "images")
        images = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        if len(images) != 1:
            raise HTTPException(status_code=400, detail="Expected exactly one image in the task directory")
        image_path = os.path.join(image_dir, images[0])

        # 读取原图
        original_image = cv2.imread(image_path)

        # 读取分割结果
        prediction = np.load(os.path.join(task_dir, 'prediction.npy'))

        # 检查点击坐标是否在有效范围内
        rows, cols = prediction.shape
        if x < 0 or x >= rows or y < 0 or y >= cols:
            raise IndexError(f"Index ({x}, {y}) out of bounds for array with shape ({rows}, {cols}).")

        # 获取点击位置的分割标签
        label = prediction[x, y]

        # 创建彩色分割图像，只高亮显示点击的区域
        colored_mask = np.zeros_like(original_image)
        if label != 0:  # 排除背景
            mask = (prediction == label).astype(np.uint8)
            color = np.array([0, 255, 0], dtype=np.uint8)  # 高亮颜色为绿色
            colored_mask[mask == 1] = color

        # 叠加图像
        alpha = 0.5
        overlay = cv2.addWeighted(original_image, 1 - alpha, colored_mask, alpha, 0)

        # 保存叠加后的图像到临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_file.name, overlay)

        return FileResponse(temp_file.name, filename="highlighted.png")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error highlighting segment: {str(e)}")

# 新增接口：生成单个属性的直方图
@app.get("/tasks/{task_id}/image/histogram")
async def get_attribute_histogram(task_id: str, attribute: str):
    try:
        # 构建 regionprops.npy 文件的完整路径
        task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id)
        regionprops = np.load(os.path.join(task_dir, 'regionprops.npy'), allow_pickle=True)

        # 提取指定属性的数据
        attribute_values = [prop[attribute] for prop in regionprops if attribute in prop]

        if not attribute_values:
            raise HTTPException(status_code=400, detail=f"Attribute '{attribute}' not found in regionprops.")

        # 生成直方图
        plt.figure()
        plt.hist(attribute_values, bins=20, edgecolor='black')
        plt.title(f'Histogram of {attribute}')
        plt.xlabel(attribute)
        plt.ylabel('Frequency')

        # 保存直方图为临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(temp_file.name)
        plt.close()

        return FileResponse(temp_file.name, filename=f"{attribute}_histogram.png")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File 'regionprops.npy' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating histogram: {str(e)}")

# 新增接口：生成属性分析的散点图
@app.get("/tasks/{task_id}/image/scatterplot")
async def get_attribute_scatterplot(task_id: str, attribute_x: str, attribute_y: str):
    try:
        # 构建 regionprops.npy 文件的完整路径
        task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id)
        regionprops = np.load(os.path.join(task_dir, 'regionprops.npy'), allow_pickle=True)

        # 提取指定属性的数据
        x_values = [prop[attribute_x] for prop in regionprops if attribute_x in prop]
        y_values = [prop[attribute_y] for prop in regionprops if attribute_y in prop]

        if not x_values or not y_values:
            raise HTTPException(status_code=400, detail=f"One or both attributes '{attribute_x}' and '{attribute_y}' not found in regionprops.")

        # 生成散点图
        plt.figure()
        plt.scatter(x_values, y_values)
        plt.title(f'Scatter Plot of {attribute_x} vs {attribute_y}')
        plt.xlabel(attribute_x)
        plt.ylabel(attribute_y)

        # 保存散点图为临时文件
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(temp_file.name)
        plt.close()

        return FileResponse(temp_file.name, filename=f"{attribute_x}_vs_{attribute_y}_scatterplot.png")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File 'regionprops.npy' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating scatter plot: {str(e)}")




@app.get("/tasks/{task_id}/image/info")
async def get_image_info(task_id: str):
    task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id, "images")
    
    if not os.path.isdir(task_dir):
        raise HTTPException(status_code=404, detail="Task directory not found")

    images = [f for f in os.listdir(task_dir) if os.path.isfile(os.path.join(task_dir, f))]
    
    if len(images) != 1:
        raise HTTPException(status_code=400, detail="Expected exactly one image in the task directory")
    
    image_path = os.path.join(task_dir, images[0])

    with Image.open(image_path) as img:
        width, height = img.size

    return {
        "filename": images[0],
        "width": width,
        "height": height
    }

# 定义一个接口来读取 prediction.npy 的指定位置值
@app.get("/predictions/value")
async def get_prediction_value(
    task_id: str,
    x: int = Query(..., description="X coordinate (row index)"),
    y: int = Query(..., description="Y coordinate (column index)")
):
    try:
        # 构建 prediction.npy 文件的完整路径
        task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id)
        predictions = np.load(os.path.join(task_dir, 'prediction.npy'))
        regionprops = np.load(os.path.join(task_dir, 'regionprops.npy'), allow_pickle=True)
        
        # 检查数组是否为二维
        if predictions.ndim != 2:
            raise ValueError("The loaded data is not a 2D array.")
        
        # 获取数组的形状
        rows, cols = predictions.shape
        
        # 验证 x 和 y 是否在有效范围内
        if x < 0 or x >= rows or y < 0 or y >= cols:
            raise IndexError(f"Index ({x}, {y}) out of bounds for array with shape ({rows}, {cols}).")
        
        label = int(predictions[x, y])
        attribute = regionprops[label]
        # 返回指定位置的值
        return {"label": label, "attribute": attribute}
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File 'prediction.npy' not found.")
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading the file: {str(e)}")

@app.get("/stats/")
async def get_statistics(task_id: str):
    """
    读取 all_stats.npy 文件并返回其内容。
    """
    try:
        # 构建 prediction.npy 文件的完整路径
        task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id)

        # 加载 all_stats.npy 文件
        all_stats = np.load(os.path.join(task_dir, "all_stats.npy"), allow_pickle=True).item()
        return {"all_stats": all_stats}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File 'all_stats.npy' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading the file: {str(e)}")
    
@app.get("/download-csv")
def download_csv(task_id: str):
    """
    将 regionprops.npy 转换为 CSV 格式并提供下载。
    """
    try:
        # 构建 prediction.npy 文件的完整路径
        task_dir = os.path.join(TASKS_ROOT_DIR_PATH, task_id)

        # 使用 NumPy 加载 .npy 文件
        data = np.load(os.path.join(task_dir, "regionprops.npy"), allow_pickle=True)

        # 检查数据是否是包含字典的数组
        if not isinstance(data, np.ndarray) or not all(isinstance(item, dict) for item in data):
            return {"error": "Data is not an array of dictionaries."}

        # 将 NumPy 数组中的字典转换为 Pandas DataFrame
        data_list = data.tolist()
        df = pd.DataFrame(data_list)

        # 将 DataFrame 转换为 CSV 格式的字符串
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()

        # 返回 CSV 文件供下载
        headers = {
            "Content-Disposition": 'attachment; filename="all_stats.csv"',
            "Content-Type": "text/csv",
        }
        return Response(content=csv_content, headers=headers)
    except Exception as e:
        return {"error": f"Failed to load or process the file: {str(e)}"}












if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)