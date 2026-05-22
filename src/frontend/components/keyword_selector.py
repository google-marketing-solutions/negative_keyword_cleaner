"""Module for handling keyword selection and filtering UI."""

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

import time
import pandas as pd
import streamlit as st
from utils import data_helper
from utils import event_helper


def load_customers_and_campaigns(state_manager):
  """Loads customers and negative keywords, and handles filtering."""
  _load_customers(state_manager)
  _load_negative_keywords(state_manager)
  _filter_campaigns(state_manager)
  _handle_filters_ready(state_manager)


def _format_customer_id(cid: int) -> str:
  """Format the customer ID into a more readable string format."""
  str_cid = str(cid)
  return f"{str_cid[:3]}-{str_cid[3:6]}-{str_cid[6:]}"


def _load_customers(state_manager):
  """Loads and displays a list of customers for user selection."""
  with st.expander(
      "2. Load Customers",
      expanded=state_manager.get("load_customers_open", True),
  ):
    df = data_helper.load_customers(st.session_state.config.login_customer_id)
    st.multiselect(
        "Selected Customers",
        df["customer_id"].apply(_format_customer_id)
        + " | "
        + df["customer_name"].astype(str),
        [],
        on_change=event_helper.handle_selected_customers,
        key="selected_customers",
    )

    if not state_manager.get("customers_ready", False):
      st.button(
          "Continue with these customers",
          on_click=event_helper.handle_continue_with_customers,
      )
      st.stop()


def _load_negative_keywords(state_manager):
  """Loads and displays negative keywords associated with the customers."""
  with st.expander(
      "3. Load negative keywords",
      expanded=state_manager.get("load_keywords_open", True),
  ):
    selected_customers = state_manager.get("selected_customers", [])
    df = data_helper.load_keywords(selected_customers).query('keyword != ""')
    state_manager.set("df_keywords", df)
    number_of_neg_kw = f"{len(df):.0f}"
    st.success(
        f"I've loaded {number_of_neg_kw} negative keywords from all "
        "campaigns. Filter only the relevant campaigns!",
        icon="🎓",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total negative keywords",
        f"{len(df):.0f}".replace(",", " "),
    )
    col2.metric(
        "Total unique keywords",
        f"{df.keyword.nunique():.0f}".replace(",", " "),
    )
    col3.metric(
        "Total campaigns",
        f"{df.campaign_id.nunique():.0f}".replace(",", " "),
    )


def _filter_campaigns(state_manager):
  """Filters negative keywords based on selected campaigns."""
  df = state_manager.get("df_keywords", pd.DataFrame())
  with st.expander(
      "4. Filter on campaigns",
      expanded=state_manager.get("filter_campaigns_open", True),
  ):
    st.multiselect(
        "Selected Campaigns",
        df.groupby(["campaign_name"])["keyword"]
        .count()
        .reset_index(name="count")
        .sort_values(["count"], ascending=True),
        [],
        on_change=event_helper.handle_selected_campaigns,
        key="selected_campaigns",
    )

    filtered_keywords = df.copy()
    if state_manager.get("selected_campaigns", None):
      selected_campaigns = state_manager.get("selected_campaigns")
      campaigns_filter = filtered_keywords["campaign_name"].isin(
          selected_campaigns
      )
      filtered_keywords = filtered_keywords[campaigns_filter]
      state_manager.set("filtered_keywords", filtered_keywords)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Selected negative keywords",
        f"{len(filtered_keywords):.0f}".replace(",", " "),
    )
    col2.metric(
        "Selected Unique keywords",
        f"{filtered_keywords.keyword.nunique():.0f}".replace(",", " "),
    )
    col3.metric(
        "Selected campaigns",
        f"{filtered_keywords.campaign_id.nunique():.0f}".replace(",", " "),
    )


def _handle_filters_ready(state_manager):
  if not state_manager.get("filters_ready", False):
    st.button(
        "Continue with these filters",
        on_click=event_helper.handle_continue_with_filters,
    )
    st.stop()

  if state_manager.get("load_keywords_open"):
    state_manager.set("load_keywords_open", False)
    time.sleep(0.05)
    st.rerun()
