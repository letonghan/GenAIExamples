# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


export ip_address=$(hostname -I | awk '{print $1}')
export no_proxy=${no_proxy}

export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export EMBEDDING_MODEL_ID="BAAI/bge-large-zh-v1.5"
export TEI_EMBEDDING_ENDPOINT="http://${ip_address}:3001"
export RERANK_MODEL_ID="BAAI/bge-reranker-base"
export TEI_RERANKING_ENDPOINT="http://${ip_address}:3004"
export GOOGLE_API_KEY=${GOOGLE_API_KEY}
export GOOGLE_CSE_ID=${GOOGLE_CSE_ID}
export LOGFLAG="true"
export INDEX_NAME="rag-redis"
export REDIS_URL="redis://${ip_address}:6379"


docker compose -f compose.yaml up -d
