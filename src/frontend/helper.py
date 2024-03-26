# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st

from utils.config import is_cloudrun, Config
from utils.event_helper import SessionStateManager


def customize_css():
    st.markdown("""
        <style >
            .stDownloadButton, div.stButton {text-align:center}
        </style>""", unsafe_allow_html=True)


def initialize_session_state(state_manager: SessionStateManager):
    state_manager.initialize("valid_config", False)
    state_manager.initialize("valid__api_config", False)
    state_manager.initialize("valid_ads_config", False)
    state_manager.initialize("valid_general_config", False)
    state_manager.initialize("updating_config", None)
    state_manager.initialize("loaded_kws", [])
    if "config" not in state_manager.list_keys():
        if is_cloudrun():
            state_manager.set("config", Config.from_gcs())
        else:
            state_manager.set("config", Config.from_disk())
    if "batch_size" not in state_manager.list_keys():
        batch_size = state_manager.get("config").batch_size
        state_manager.set("batch_size", batch_size)
