FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[app,api,parallel]"

COPY app ./app
COPY configs ./configs

# Default: the REST API. Override CMD for the Streamlit app:
#   docker run -p 8501:8501 expkit streamlit run app/streamlit_app.py --server.address 0.0.0.0
EXPOSE 8000 8501
CMD ["uvicorn", "expkit.server:app", "--host", "0.0.0.0", "--port", "8000"]
