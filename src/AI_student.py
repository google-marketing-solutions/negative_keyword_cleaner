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

try:
    # Loads local env variables.
    from dotenv import find_dotenv, load_dotenv
    env_file = find_dotenv()
    load_dotenv(env_file)
except ImportError as err:
    print("ImportError:", err)
except Exception as err:
    print(f"Failed to import local env variables: {err}")

import streamlit as st

import frontend.helper as st_helper
from frontend import settings_ui
from frontend import main_ui

# The Page UI starts here
st.set_page_config(
    page_title="AI Student",
    page_icon="🎓",
    layout="centered"
)

st_helper.initialize_session_state()
st_helper.customize_css()
settings_ui.validate_setup()

if st.session_state.valid_config:
    main_ui.display_page()
else:
    settings_ui.display_page()
