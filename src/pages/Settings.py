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

"""This script initializes the Streamlit-based user interface for the app.
"""
import streamlit as st

import frontend.helper as st_helper
import frontend.settings_ui as settings_ui
from utils.event_helper import session_state_manager

with open("src/frontend/style/style.css") as f:
  st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

state_manager = session_state_manager()

st_helper.customize_css()
settings_ui.validate_setup()
settings_ui.display_page()
