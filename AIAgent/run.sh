# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


export ip_address=$(hostname -I | awk '{print $1}')
export no_proxy=${ip_address}
export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export recursion_limit_worker=12
export llm_endpoint_url="http://${ip_address}:8085"
export model="Qwen/Qwen2.5-72B-Instruct"
export LOGFLAG="true"
export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:3001"
export WEB_RETRIEVER_ENDPOINT="http://${ip_address}:3003"
export TEI_RERANKING_ENDPOINT="http://${ip_address}:3004"


function start_tgi_ragagent() {
    export HF_CACHE_DIR="./data"
    export NUM_SHARDS=2

    docker compose -f compose.yaml up -d
}

python ai_agent.py

