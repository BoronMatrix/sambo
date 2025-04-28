from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import shutil
import os

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

from micro_sam.evaluation.model_comparison import _enhance_image
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

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


def calculate_statistics(values):
    """
    计算统计值：均值、标准差、最大值、最小值、P0、P10、P50、P90、P100。
    """
    values = np.array(values)
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
    return stats

app = FastAPI()

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

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # 检查文件是否为图片（可选）
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"message": "文件不是图片"})

    # 保存上传的图片到指定目录
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    # with open(file_path, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    
    # 原始文档用file.file, 要保存用file_path
    prediction = run_automatic_instance_segmentation(file.file, ndim=2, model_type=model_choice)

    # 保存为二进制文件
    np.save('prediction.npy', prediction)

    return {"message": "图片上传成功", "file_path": file_path}



# 定义一个接口来读取 prediction.npy 的指定位置值
@app.get("/predictions/value")
async def get_prediction_value(
    x: int = Query(..., description="X coordinate (row index)"),
    y: int = Query(..., description="Y coordinate (column index)")
):
    try:
        # 加载 prediction.npy 文件
        predictions = np.load("prediction.npy")
        
        # 检查数组是否为二维
        if predictions.ndim != 2:
            raise ValueError("The loaded data is not a 2D array.")
        
        # 获取数组的形状
        rows, cols = predictions.shape
        
        # 验证 x 和 y 是否在有效范围内
        if x < 0 or x >= rows or y < 0 or y >= cols:
            raise IndexError(f"Index ({x}, {y}) out of bounds for array with shape ({rows}, {cols}).")
        
        # 返回指定位置的值
        return {"value": float(predictions[x, y])}
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File 'prediction.npy' not found.")
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading the file: {str(e)}")
    

@app.get("/predictions/regionprops", response_model=List[Dict])
async def get_individual_properties():
    try:
        # 加载 prediction.npy 文件
        predictions = np.load("prediction.npy")
        
        # 确保数据是二维数组
        if predictions.ndim != 2:
            raise ValueError("The loaded data is not a 2D array.")
        
        # 计算每个区域的属性
        properties = calculate_region_properties(predictions)
        
        return properties
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File 'prediction.npy' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")


@app.get("/predictions/regionprops/stats", response_model=Dict[str, Dict])
async def get_aggregate_statistics():
    try:
        # 加载 prediction.npy 文件
        predictions = np.load("prediction.npy")
        
        # 确保数据是二维数组
        if predictions.ndim != 2:
            raise ValueError("The loaded data is not a 2D array.")
        
        # 计算每个区域的属性
        properties = calculate_region_properties(predictions)
        
        # 提取所有属性值
        all_stats = {}
        for key in properties[0].keys():  # 遍历属性名称
            values = [prop[key] for prop in properties]
            all_stats[key] = calculate_statistics(values)
        
        return all_stats
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File 'prediction.npy' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")
