# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


export ip_address=$(hostname -I | awk '{print $1}')
export no_proxy=${ip_address}

export HUGGINGFACEHUB_API_TOKEN=${HUGGINGFACEHUB_API_TOKEN}
export IMAGE_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"
export HABANA_VISIBLE_DEVICES="5"


docker compose -f compose.yaml up -d

