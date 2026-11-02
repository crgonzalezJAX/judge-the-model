#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ! -f "streamlit_app.py" ]]; then
  echo "Error: streamlit_app.py is missing from the project folder." >&2
  exit 1
fi

if [[ -x ".venv/Scripts/python.exe" ]]; then
  python_cmd=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  python_cmd=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_cmd="$(command -v python)"
else
  echo "Error: Python is unavailable. Create the project .venv or activate a Python environment." >&2
  exit 1
fi

if ! "$python_cmd" -c "import streamlit" >/dev/null 2>&1; then
  echo "Error: Streamlit is unavailable in $python_cmd. Install requirements.txt into that environment." >&2
  exit 1
fi

echo "Opening address: http://localhost:8501/"
echo "Press Ctrl+C to stop the app."
exec "$python_cmd" -m streamlit run "streamlit_app.py"
