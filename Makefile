PY=python3
PIP=pip

setup:
	$(PIP) install -r requirements.txt

lint:
	ruff check .

format:
	black .

unit:
	pytest -q tests/unit

all-tests:
	pytest -q

run:
	streamlit run app/streamlit_app.py
