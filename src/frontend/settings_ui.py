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
"""This module manages the config and validation of the app's setup.
"""

import logging

import streamlit as st

from utils import auth
from utils import event_helper
from utils.config import is_cloudrun

logger = logging.getLogger("streamlit")
state_manager = event_helper.session_state_manager()

_CONFIG_CREDENTIALS = 1
_CONFIG_ADS = 2
_CONFIG_AI_API = 3
_CONFIG_GENERAL = 4

DEV_TOKEN_HELP = """Refer to
    [How to obtain a developer token](https://developers.google.com/google-ads/api/docs/first-call/dev-token#:~:text=A%20developer%20token%20from%20Google,SETTINGS%20%3E%20SETUP%20%3E%20API%20Center.)
    for more information"""

GOOGLE_ADS_HELP = """Google Ads MCC account ID. You can set
    it both with or without hyphens (ie. XXX-XXX-XXXX)"""


def validate_setup() -> None:
  """Validates the application setup by checking various configuration params.

  Updates the session state to reflect the validity of different
  configuration sections.
  """
  config = state_manager.get("config")
  state_manager.set("valid_ads_config", False)
  state_manager.set("valid_api_config", False)
  state_manager.set("valid_general_config", False)

  # Validate Ads Settings
  if config.login_customer_id and config.developer_token:
    state_manager.set("valid_ads_config", True)

  # Validate API Settings
  if config.google_api_key or config.openai_api_key:
    state_manager.set("valid_api_config", True)

  # Validate Tool Settings
  if config.batch_size:
    state_manager.set("valid_general_config", True)

  config_updated = False

  # TODO: Validate the access token and refresh it if needed

  # Save any changes in the config file
  if config_updated:
    _save_config(config)

  is_valid_config = all([
      state_manager.get("valid_ads_config"),
      state_manager.get("valid_api_config"),
      state_manager.get("valid_general_config"),
  ])
  state_manager.set("valid_config", is_valid_config)


def _save_config(config) -> None:
  """Saves the provided config to the appropriate storage based on the env.

  Args:
    config: The configuration object to be saved.
  """
  if is_cloudrun():
    config.save_to_gcs()
  else:
    config.save_to_disk()


def update_config(updating_config: str) -> None:
  """Updates the configuration section currently being modified.

  Args:
    updating_config (str): The configuration section being updated.
  """
  state_manager.set("updating_config", updating_config)
  state_manager.set("valid_config", False)


def save_api_config(config) -> None:
  """Saves the API configuration for LLM and validates the setup.

  Args:
    config: The configuration object containing API settings.
  """
  _save_config(config)
  st.session_state.valid_api_config = True
  st.session_state.updating_config = False
  validate_setup()


def save_general_config(config) -> None:
  """Saves the general application settings and validates the setup.

  Args:
    config: The configuration object containing general settings.
  """
  config.batch_size = st.session_state.batch_size
  _save_config(config)
  if st.session_state.batch_size:
    st.session_state.valid_general_config = True
    st.session_state.updating_config = False
  validate_setup()


def display_page() -> None:
  """Display the application settings page.
  """
  st.header("App Settings")

  auth.authenticate_user()

  if state_manager.get("valid_config"):
    st.success("Application successfully setup ✅")

  config = state_manager.get("config")

  modify_general_config = any([
      not st.session_state.valid_general_config,
      st.session_state.updating_config == _CONFIG_GENERAL,
  ])

  with st.expander("**General Settings**", modify_general_config):
    with st.form("Tool"):
      if all([
          not st.session_state.updating_config,
          not st.session_state.valid_general_config,
      ]):
        st.error("Incorrect tool configuration", icon="⚠️")

      st.number_input(
          "Batch size",
          min_value=0,
          max_value=20,
          value=config.batch_size,
          key="batch_size",
          step=1,
          help="Number of keywords to review per batch",
          disabled=not modify_general_config,
      )

      if modify_general_config:
        st.form_submit_button(
            "Save",
            on_click=save_general_config,
            args=[st.session_state.config],
        )
      else:
        st.form_submit_button(
            "Edit", on_click=update_config, args=[_CONFIG_GENERAL]
        )
