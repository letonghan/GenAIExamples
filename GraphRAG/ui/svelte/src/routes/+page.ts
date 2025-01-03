//  Copyright (C) 2024 Intel Corporation
//  SPDX-License-Identifier: Apache-2.0

import { browser } from "$app/environment";
import { LOCAL_STORAGE_KEY } from "$lib/shared/constant/Interface";

export const load = async () => {
	if (browser) {
		const chat = localStorage.getItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY);

		return {
			chatMsg: JSON.parse(chat || "[]"),
		};
	}
};
