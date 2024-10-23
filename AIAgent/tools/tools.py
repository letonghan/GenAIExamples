# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import requests


def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for a specific query."""
    # use worker agent (DocGrader) to search the knowledge base
    url = os.environ.get("WORKER_AGENT_URL")
    print(url)
    proxies = {"http": ""}
    payload = {
        "query": query,
    }
    response = requests.post(url, json=payload, proxies=proxies)
    return response.json()["text"]

