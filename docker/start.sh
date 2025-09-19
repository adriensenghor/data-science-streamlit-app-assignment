#!/usr/bin/env bash
set -euo pipefail
exec streamlit run app/streamlit_app.py --server.headless=true --server.port=${PORT:-8501}
