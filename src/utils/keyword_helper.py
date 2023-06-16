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

import re
from enum import Enum
from gaarf.api_clients import GoogleAdsApiClient
from gaarf.query_executor import AdsReportFetcher, GaarfReport
from gaarf.utils import get_customer_ids
from utils.config import Config
from utils.gaarf_queries import AdgroupNegativeKeywords, CampaignNegativeKeywords, KeywordLevel

_GOOGLE_ADS_API_VERSION = "v13"


class MatchType(Enum):
    BROAD = 'BROAD'
    EXACT = 'EXACT'
    PHRASE = 'PHRASE'
    UNKNOWN = 'UNKNOWN'
    UNSPECIFIED = 'UNSPECIFIED'


class Keyword:
    def __init__(
        self, criterion_id:str, keyword:str, is_negative:bool,
        match_type:MatchType, level:KeywordLevel, adgroup_id:int,
        adgroup_name:str, campaign_id:int, campaign_name:str, account_id:int,
        account_name:str):
        self.criterion_id = criterion_id
        self.kw_text = keyword
        self.is_negative = is_negative
        self.match_type = match_type
        self.level = level
        self.adgroup_id = adgroup_id
        self.adgroup_name = adgroup_name
        self.campaign_id = campaign_id
        self.campaign_name = campaign_name
        self.account_id = account_id
        self.account_name = account_name

    def get_clean_keyword_text(self):
        match self.match_type:
            case MatchType.BROAD:
                return self.kw_text.replace('+', '')
            case MatchType.PHRASE:
                return self.kw_text.replace("\"", "")
            case MatchType.EXACT:
                return self.kw_text.replace("[", "").replace("]","")
        return self.kw_text


class KeywordHelper:
    def __init__(self, config:Config):
        # Expand the mcc account to child accounts to initialize the report fetcher
        googleads_api_client = GoogleAdsApiClient(
            config_dict=config.__dict__,
            version=_GOOGLE_ADS_API_VERSION)
        try:
            customer_ids = get_customer_ids(
                googleads_api_client,
                config.login_customer_id)
            self.report_fetcher = AdsReportFetcher(
                googleads_api_client,
                customer_ids)
        except Exception.InternalServerError as e:
            return None

    def get_neg_keywords(self) -> GaarfReport:
        adgroup_neg_kws = self.report_fetcher.fetch(AdgroupNegativeKeywords())
        campaign_neg_kws = self.report_fetcher.fetch(CampaignNegativeKeywords())
        return adgroup_neg_kws + campaign_neg_kws

    def clean_and_dedup(self, raw_keyword_data: GaarfReport) -> dict:
        all_keywords = {}
        for kw in raw_keyword_data:
            keyword_obj = Keyword(
                kw.criterion_id, kw.keyword, kw.is_negative, kw.match_type,
                kw.level, kw.adgroup_id, kw.adgroup_name, kw.campaign_id,
                kw.campaign_name, kw.account_id, kw.account_name)
            clean_kw = keyword_obj.get_clean_keyword_text()
            if clean_kw not in all_keywords:
                all_keywords[clean_kw] = [keyword_obj]
            else:
                all_keywords[clean_kw].append(keyword_obj)

        return all_keywords
