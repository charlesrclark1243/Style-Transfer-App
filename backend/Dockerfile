FROM python:3.12-slim

# Keep the venv outside /app so the dev bind mount doesn't hide it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

EXPOSE 8000

CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
