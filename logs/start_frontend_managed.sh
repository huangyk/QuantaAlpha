#!/bin/bash
cd "D:/qqmt/QuantaAlpha/frontend-v2"
export PATH="D:/nvm4w/nodejs:$PATH"
nohup npm run dev -- --host 0.0.0.0 --port 3000 > ../logs/web_frontend_managed.log 2>&1 &
echo "frontend started pid=$!"
wait
