import cv2
import tempfile
import os
from fastapi import FastAPI
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

# micro_sam
import os
from glob import glob
from typing import Optional, Union, Tuple
from PIL import Image
from typing import List
from skimage import measure
from typing import List, Dict

import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label as connected_components
from scipy.stats import scoreatpercentile

import torch

#from micro_sam.evaluation.model_comparison import _enhance_image
#from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from pydantic import BaseModel
import uuid
import json
import matplotlib.pyplot as plt



# 创建一个目录用于保存上传的图片
UPLOAD_DIR = "uploads"
MODEL_DIR = "models/vit_b_lm"
SEGMENT_DIR = "segmentations"
model_choice = "vit_b_lm"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SEGMENT_DIR, exist_ok=True)
# 任务存储的根目录路径
TASKS_ROOT_DIR_PATH = "tasks"


#app = FastAPI(docs_url=None, redoc_url=None)
app = FastAPI()

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





