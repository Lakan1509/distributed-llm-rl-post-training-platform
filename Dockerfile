FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
CMD ["python", "-m", "rl_platform.cli", "--prompts", "32", "--workers", "4"]
