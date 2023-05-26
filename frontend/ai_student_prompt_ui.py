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


def get_neg_keywords(config):
    # TODO: extract the negative keywords directly from here once streamlit 1.23 is released
    # https://docs.streamlit.io/library/changelog
    # https://github.com/streamlit/streamlit/pull/6622

    # googleads_api_client = GoogleAdsApiClient(
    # config_dict=config.__dict__,
    # version=_GOOGLE_ADS_API_VERSION)
    # report_fetcher = AdsReportFetcher(
    #     googleads_api_client,
    #     [config.login_customer_id])
    # adgroup_neg_kws = report_fetcher.fetch(AdgroupNegativeKeywords())
    # campaign_neg_kws = report_fetcher.fetch(CampaignNegativeKeywords())

    # neg_keywords_report = adgroup_neg_kws + campaign_neg_kws
    # displayable_report = neg_keywords_report[['keyword',
    #                                         'match_type',
    #                                         'level',
    #                                         'adgroup_name',
    #                                         'campaign_name',
    #                                         'account_name']]
    pass


def display_page():
    config = st.session_state.config
    with st.expander("**Prompt**", expanded=st.session_state.valid_config):
        st.button(
            "Get Negative Keywords",
            key="get_neg_kws",
            on_click=get_neg_keywords,
            args=[config])
        st.text_area(label="Tell us about your business")
