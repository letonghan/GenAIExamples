# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

WORKPATH=$(dirname "$PWD")/..
echo "WORKDIR=${WORKDIR}"
export ip_address=$(hostname -I | awk '{print $1}')
export no_proxy=${ip_address}
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export TOOLSET_PATH="/home/kding1/letong/ai_agent/tools/"
export recursion_limit_worker=12
export llm_endpoint_url="http://${ip_address}:8085"
export model="Qwen/Qwen2.5-72B-Instruct"
export temperature=0.01
export max_new_tokens=512
export LOGFLAG="true"


function start_tgi_ragagent() {
    export HF_CACHE_DIR="/home/kding1/letong/data"
    export NUM_SHARDS=2

    docker compose -f compose.yaml up -d
}

start_tgi_ragagent


function start_agent_planner() {
    export strategy=rag_agent
    export recursion_limit=12
    export streaming=false
    export require_human_feedback=false
    export port=9090
    export WORKER_AGENT_URL="http://${ip_address}:9095/v1/chat/completions"
    export tool_yaml_path="/home/kding1/letong/ai_agent/tools/agent_tools.yaml"

    python agent_planner.py
}

start_agent_planner