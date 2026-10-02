#!/bin/bash
cd "D:/qqmt/QuantaAlpha/frontend-v2"
nohup bash -c 'CONDA_DEFAULT_ENV=quantaalpha PYTHONUTF8=1 D:/Users/lenovo/miniconda3/envs/quantaalpha/python.exe backend/app.py' > ../logs/web_backend_managed.log 2>&1 &
echo "backend started pid=$!"
wait
