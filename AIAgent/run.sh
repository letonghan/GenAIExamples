# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

export ip_address=$(hostname -I | awk '{print $1}')
export PYTHONPATH="/home/kding1/letong/GenAIComps/"
export strategy=rag_agent
export recursion_limit=12
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export llm_endpoint_url="http://${ip_address}:8085"
export model="meta-llama/Meta-Llama-3.1-70B-Instruct"
export temperature=0.01
export max_new_tokens=512
export streaming=false
export require_human_feedback=false
export no_proxy=${ip_address}
export LOGFLAG="true"
export port=9090
export WORKER_AGENT_URL="http://${ip_address}:9095/v1/chat/completions"
export CRAG_SERVER="http://${ip_address}:8080"
export PYTHONPATH=/"home/kding1/letong/GenAIComps"

python agent_planner.py
