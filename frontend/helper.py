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
from utils.config import Config


def customize_css():
    st.markdown("""
        <style >
            .stDownloadButton, div.stButton {text-align:center}
        </style>""", unsafe_allow_html=True)


def initialize_session_state():
    if "valid_config" not in st.session_state:
        st.session_state.valid_config = False
    if "valid_credentials" not in st.session_state:
        st.session_state.valid_credentials = False
    if "valid_ads_config" not in st.session_state:
        st.session_state.valid_ads_config = False
    if "valid_api_config" not in st.session_state:
        st.session_state.valid_api_config = False
    if "updating_config" not in st.session_state:
        st.session_state.updating_config = None
    if "config" not in st.session_state:
        st.session_state.config = Config()
    if "loaded_kws" not in st.session_state:
        st.session_state.loaded_kws = []


def display(value):
    if value:
        return value
    else:
        return ''
