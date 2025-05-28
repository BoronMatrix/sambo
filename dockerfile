# 使用 Miniconda 作为基础镜像
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /app

# 安装系统级依赖（OpenGL）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libegl1 \
        libgles2 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 拷贝 Python 应用代码到容器中
COPY . .

# 创建 Conda 环境
RUN conda env create -f environment.yaml

# 暴露端口 8000
EXPOSE 8000

# 启动 FastAPI 应用
CMD ["conda", "run", "-n", "sam", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]