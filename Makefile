PY=python3
PIP=pip

# Install dependencies
setup:
	$(PIP) install -r requirements.txt

# Run the Streamlit app
run:
	streamlit run app/streamlit_app.py

# Run tests
test:
	pytest tests/ -v

# Code quality
lint:
	ruff check .

format:
	black .

# Docker commands
docker-build:
	docker build -t bike-counter-app .

docker-run:
	docker run -p 8501:8501 bike-counter-app

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
