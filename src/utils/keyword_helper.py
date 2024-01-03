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

from collections.abc import MutableSequence
from enum import Enum
import re
from typing import Sequence, Union

from gaarf.api_clients import GoogleAdsApiClient
from gaarf.query_editor import QuerySpecification
from gaarf.query_executor import AdsReportFetcher, GaarfReport
from gaarf.report import GaarfRow
from google.api_core import exceptions

from utils import auth
from utils.config import Config
from utils.gaarf_queries import CustomerNames, AdgroupNegativeKeywords, CampaignNegativeKeywords, KeywordLevel

_GOOGLE_ADS_API_VERSION = "v13"


class MatchType(Enum):
    BROAD = 'BROAD'
    EXACT = 'EXACT'
    PHRASE = 'PHRASE'
    UNKNOWN = 'UNKNOWN'
    UNSPECIFIED = 'UNSPECIFIED'


def get_customer_ids(ads_client: GoogleAdsApiClient,
                     customer_id: Union[str, MutableSequence[str]],
                     customer_ids_query: str = None) -> Sequence[str]:
    """Gets list of customer_ids from an MCC account.

    Args:
        ads_client: GoogleAdsApiClient used for connection.
        customer_id: MCC account_id.
        custom_query: GAQL query used to reduce the number of customer_ids.
    Returns:
        All customer_ids from MCC safisfying the condition.
    """

    # Fetches ENABLED and CANCELED accounts.
    query = """
    SELECT customer_client.id
    FROM customer_client
    WHERE customer_client.manager = FALSE AND
    customer_client.status = 'ENABLED' AND
    customer.status = 'ENABLED'
    """

    query_specification = QuerySpecification(query).generate()
    if not isinstance(customer_id, MutableSequence):
        customer_id = customer_id.split(",")
    report_fetcher = AdsReportFetcher(ads_client, customer_id)
    customer_ids = report_fetcher.fetch(query_specification).to_list()
    if customer_ids_query:
        report_fetcher = AdsReportFetcher(ads_client, customer_ids)
        query_specification = QuerySpecification(customer_ids_query).generate()
        customer_ids = report_fetcher.fetch(query_specification)
        customer_ids = [
            row[0] if isinstance(row, GaarfRow) else row
            for row in customer_ids
        ]

    customer_ids = list(
        set([customer_id for customer_id in customer_ids if customer_id != 0]))

    return customer_ids

class Customer:
    def __init__(self, customer_id:str, customer_name:str):
        self.customer_id = customer_id
        self.customer_name = customer_name

class Keyword:
    def __init__(
        self, criterion_id:str, original_keyword:str, keyword:str, 
        is_negative:bool, match_type:MatchType, level:KeywordLevel, 
        adgroup_id:int, adgroup_name:str, campaign_id:int, 
        campaign_name:str, account_id:int, account_name:str):
        self.criterion_id = criterion_id
        self.original_kw_text = original_keyword
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
            config_dict={
                'developer_token': config.developer_token,
                'login_customer_id': config.login_customer_id,
                'use_proto_plus': config.use_proto_plus,
                'client_id': auth.OAUTH_CLIENT_ID,
                'client_secret': auth.OAUTH_CLIENT_SECRET,
                'refresh_token': auth.get_access_token(),
            },
            version=_GOOGLE_ADS_API_VERSION)
        try:
            customer_ids = get_customer_ids(
                googleads_api_client,
                config.login_customer_id)
            self.report_fetcher = AdsReportFetcher(
                googleads_api_client,
                customer_ids)
        except exceptions.InternalServerError as e:
            return None

    def get_customers(self) -> GaarfReport:
        customers = self.report_fetcher.fetch(CustomerNames())
        return customers

    def get_neg_keywords(self,selected_customers:list) -> GaarfReport:
        pattern = r'^(\d+)'
        customer_ids = []
        for customer in selected_customers:
            match = re.match(pattern, customer)
            if match:
                customer_ids.append(match.group(1))
        adgroup_neg_kws = self.report_fetcher.fetch(AdgroupNegativeKeywords(), customer_ids)
        campaign_neg_kws = self.report_fetcher.fetch(CampaignNegativeKeywords(), customer_ids)
        return adgroup_neg_kws + campaign_neg_kws

    def clean_and_dedup(self, raw_keyword_data: GaarfReport) -> dict:
        all_keywords = {}
        for kw in raw_keyword_data:
            keyword_obj = Keyword(
                kw.criterion_id, kw.keyword, kw.keyword, kw.is_negative, kw.match_type,
                kw.level, kw.adgroup_id, kw.adgroup_name, kw.campaign_id,
                kw.campaign_name, kw.account_id, kw.account_name)
            clean_kw = keyword_obj.get_clean_keyword_text()

            if clean_kw not in all_keywords:
                all_keywords[clean_kw] = [keyword_obj]
            else:
                all_keywords[clean_kw].append(keyword_obj)
        return all_keywords
