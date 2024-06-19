# Copyright 2024 Google LLC
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

import textwrap

import streamlit as st

from .faq import display_faq_component


def display_sidebar_component():
  with st.sidebar:
    st.markdown(textwrap.dedent("""
            # About
            Our AI Student aims at helping you maintain healthy negative keywords.

            Please share your feedback or any suggestion at negatives@google.com 💡

            _Made by Google Ads Professional Services_

            ---
            """))

    display_faq_component()
