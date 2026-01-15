#!/bin/zsh
set -e

cd "/Users/yasar/Documents/work/orchestros ai"
if [ -f "venv/bin/activate" ]; then
  source "venv/bin/activate"
fi

if [ -x "venv/bin/streamlit" ]; then
  "venv/bin/streamlit" run "scripts/gui_test_phase1_streamlit.py"
else
  streamlit run "scripts/gui_test_phase1_streamlit.py"
fi

