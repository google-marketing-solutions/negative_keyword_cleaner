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

import logging
import os
import re
import urllib.parse

import requests
import streamlit as st

import frontend.helper as st_helper
from utils import auth
from utils.config import is_appengine, Config

logger = logging.getLogger("streamlit")

_CONFIG_CREDSENTIALS = 1
_CONFIG_ADS = 2
_CONFIG_AI_API = 3

DEV_TOKEN_HELP = """Refer to
    [How to obtain a developer token](https://developers.google.com/google-ads/api/docs/first-call/dev-token#:~:text=A%20developer%20token%20from%20Google,SETTINGS%20%3E%20SETUP%20%3E%20API%20Center.)
    for more information"""

GOOGLE_ADS_HELP = """Google Ads MCC account ID. You can set 
    it both with or without hyphens (ie. XXX-XXX-XXXX)"""


def validate_setup():
    config = st.session_state.config
    st.session_state.valid_ads_config = False
    st.session_state.valid_api_config = False

    # Validate Ads Settings
    if config.login_customer_id and config.developer_token:
        st.session_state.valid_ads_config = True

    # Validate API Settings
    if config.openai_api_key or config.google_api_key:
        st.session_state.valid_api_config = True

    config_updated = False

    # TODO: Validate the access token and refresh it if needed

    # Save any changes in the config file
    if config_updated:
        save_config(config)

    st.session_state.valid_config = all([
        st.session_state.valid_ads_config,
        st.session_state.valid_api_config,
    ])


def save_config(config):
    if is_appengine():
        config.save_to_gcs()
    else:
        config.save_to_disk()


def update_config(updating_config):
    st.session_state.updating_config = updating_config
    st.session_state.valid_config = False


def save_ads_config(config):
    config.login_customer_id = str(
        st.session_state.login_customer_id.replace('-', ''))
    config.developer_token = st.session_state.developer_token
    save_config(config)
    if st.session_state.login_customer_id and st.session_state.developer_token:
        st.session_state.valid_ads_config = True
        st.session_state.updating_config = False
    validate_setup()


def save_api_config(config):
    config.openai_api_key = st.session_state.openai_api_key
    config.google_api_key = st.session_state.google_api_key
    save_config(config)
    if st.session_state.openai_api_key or st.session_state.google_api_key:
        st.session_state.valid_api_config = True
        st.session_state.updating_config = False
    validate_setup()


def display_page():
    auth.authenticate_user()

    st.subheader("App Settings")

    if st.session_state.valid_config:
        st.success("Application successfully setup ✅")

    config = st.session_state.config
    modify_ads_config = any([
        not st.session_state.valid_ads_config,
         st.session_state.updating_config == _CONFIG_ADS])
    modify_api_config = any([
        not st.session_state.valid_api_config,
        st.session_state.updating_config == _CONFIG_AI_API])

    with st.expander("**Google Ads**", expanded=modify_ads_config):
        with st.form("Google Ads"):
            if all([
                not st.session_state.updating_config,
                not st.session_state.valid_ads_config]):
                st.error(f"Google Ads configuration missing", icon="⚠️")
            st.text_input(
                "MCC ID",
                value=st_helper.display(config.login_customer_id),
                key="login_customer_id",
                disabled= not modify_ads_config,
                help="Google Ads MCC account ID. You can set it both with or without hyphens XXX-XXX-XXXX")
            st.text_input(
                "Google Ads API Developer Token",
                value=st_helper.display(config.developer_token),
                key="developer_token",
                disabled= not modify_ads_config,
                help=DEV_TOKEN_HELP)

            if modify_ads_config:
                st.form_submit_button(
                    "Save",
                    on_click=save_ads_config,
                    args=[st.session_state.config])
            else:
                st.form_submit_button(
                    "Edit", on_click=update_config,args=[_CONFIG_ADS])

    with st.expander("**Large Language Model APIs**", modify_api_config):
        with st.form("API"):
            print("st.session_state.updating_config:", st.session_state.updating_config)
            if all([
                not st.session_state.updating_config,
                not st.session_state.valid_api_config]):
                st.error(f"AI API token missing", icon="⚠️")

            st.text_input(
                "Google API Key",
                value=st_helper.display(config.google_api_key),
                key="google_api_key",
                disabled= not modify_api_config,
                type="password")

            st.text_input(
                "OpenAI API Key",
                value=st_helper.display(config.openai_api_key),
                key="openai_api_key",
                disabled= not modify_api_config,
                type="password")

            if modify_api_config:
                st.form_submit_button(
                    "Save",
                    on_click=save_api_config,
                    args=[st.session_state.config])
            else:
                st.form_submit_button(
                    "Edit", on_click=update_config, args=[_CONFIG_AI_API])
