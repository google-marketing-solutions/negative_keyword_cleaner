"""Module for handling advertiser information UI and logic."""

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

import re
from frontend import models
import requests
import streamlit as st
from utils import event_helper

_URL_REGEX = (
    r"^(https?:\/\/)"  # protocol
    r"((([a-z\d]([a-z\d-]*[a-z\d])*)\.)+[a-z]{2,}|"  # domain name
    r"((\d{1,3}\.){3}\d{1,3}))"  # OR ip (v4) address
    r"(\:\d+)?(\/[-a-z\d%_.~+]*)*"  # port and path
    r"(\?[;&a-z\d%_.~+=-]*)?"  # query string
    r"(\#[-a-z\d_]*)?$"  # fragment locator
)


def _set_manual_context(state_manager):
  def _set_manual_context_callback():
    state_manager.set("manual_context", True)
    state_manager.set("context_open", True)

  return _set_manual_context_callback


def handle_advertiser_information(state_manager, llm):
  """Handles the advertiser's information input and processes it."""
  with st.expander(
      "1. Advertiser Information",
      expanded=state_manager.get("context_open", True),
  ):
    company_homepage_url = st.text_input(
        "Company Homepage URL",
        placeholder="https://...",
        key="company_homepage_url",
        value=state_manager.get("company_homepage_url", ""),
        disabled=st.session_state.get("manual_context", False),
    )
    _process_company_homepage(company_homepage_url, state_manager, llm)
    company_pitch, exclude_pitch = _get_company_and_exclude_pitches(
        state_manager
    )

    # We need to store these in state manager so they are available for
    # prompt
    state_manager.set("company_pitch", company_pitch)
    state_manager.set("exclude_pitch", exclude_pitch)

  _handle_context_ready(state_manager)


def _process_company_homepage(company_homepage_url, state_manager, llm):
  """Processes the company's homepage URL to gather its text."""
  if not company_homepage_url and not state_manager.get("manual_context"):
    st.info(
        "Once I have their website URL, I can directly read and "
        "understand who this customer is.",
        icon="🧑‍🎓",
    )
    st.write("If you want to give me advertiser information directly")
    st.button(
        "Add context manually",
        on_click=_set_manual_context(state_manager),
    )
    st.stop()

  if company_homepage_url and not state_manager.get("manual_context"):
    if not re.match(_URL_REGEX, company_homepage_url):
      st.error(
          "The URL is not in a valid format. Please check it and try again."
      )
      st.stop()
    else:
      try:
        with st.spinner("I'm browsing their website..."):
          homepage_docs = models.fetch_landing_page_text(company_homepage_url)
          state_manager.set("homepage_fetched", True)
        st.success("Browsing done, I've collected enough info", icon="🧑‍🎓")

        with st.spinner(
            "I'm now consolidating everything into an executive summary (this"
            " will take a minute) ..."
        ):
          if not state_manager.get("homepage_summary", None):
            homepage_summary = models.summarize_text(
                homepage_docs, llm
            ).strip()
            state_manager.set("homepage_summary", homepage_summary)
          st.success(
              "Summarizing done but feel free to correct anything that I've"
              " written.",
              icon="🎓",
          )
      except requests.exceptions.RequestException as e:
        st.error(f"Could not crawl the website: {e}")
        if st.button("Add context manually"):
          state_manager.set("manual_context", True)
          st.rerun()
        st.stop()


def _get_company_and_exclude_pitches(state_manager):
  """Displays the company's executive summaries."""
  homepage_summary = state_manager.get("homepage_summary", "")
  company_pitch = st.text_area(
      "✅ [Positive prompt] Advertiser's executive summary",
      help="You can add campaign information below",
      placeholder="Describe what the company is selling in a few words",
      value=homepage_summary,
      height=150,
  )

  st.info(
      "Happy to know more about what you don't want to target ads for",
      icon="🧑‍🎓",
  )

  exclude_pitch = st.text_area(
      "❌ [Negative prompt] Exclude summary",
      placeholder=(
          "Describe what you don't want to target ads for (your brand,"
          " competitor's names, other non-relevant topics ...)"
      ),
      height=50,
  )

  return company_pitch, exclude_pitch


def _handle_context_ready(state_manager):
  if not state_manager.get("context_ready"):
    st.button(
        "Continue with this context",
        on_click=event_helper.handle_continue_with_context,
    )
    st.stop()
  elif state_manager.get("context_open"):
    state_manager.set("context_open", False)
    st.rerun()
