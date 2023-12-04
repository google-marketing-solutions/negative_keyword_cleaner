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

from enum import Enum
from gaarf.base_query import BaseQuery


class KeywordLevel(Enum):
    CAMPAIGN = 'Campaign'
    ADGROUP = 'Adgroup'


class AdgroupNegativeKeywords(BaseQuery):
    def __init__(self) -> None:
        self.query_text = f"""
            SELECT
                ad_group_criterion.criterion_id AS criterion_id,
                ad_group_criterion.keyword.text AS keyword,
                ad_group_criterion.negative AS is_negative,
                ad_group_criterion.keyword.match_type AS match_type,
                '{KeywordLevel.ADGROUP.value}' AS level,
                ad_group.id AS adgroup_id,
                ad_group.name AS adgroup_name,
                campaign.id AS campaign_id,
                campaign.name AS campaign_name,
                customer.id AS account_id,
                customer.descriptive_name AS account_name
            FROM ad_group_criterion
            WHERE
                ad_group_criterion.type = 'KEYWORD' AND
                ad_group_criterion.negative = True AND
                campaign.status = 'ENABLED' AND
                ad_group.status = 'ENABLED'
        """
class CampaignNegativeKeywords(BaseQuery):
    def __init__(self) -> None:
        self.query_text = f"""
            SELECT
                campaign_criterion.criterion_id AS criterion_id,
                campaign_criterion.keyword.text AS keyword,
                campaign_criterion.negative AS is_negative,
                campaign_criterion.keyword.match_type AS match_type,
                '{KeywordLevel.CAMPAIGN.value}' AS level,
                0 AS adgroup_id,
                '' AS adgroup_name,
                campaign.id AS campaign_id,
                campaign.name AS campaign_name,
                customer.id AS account_id,
                customer.descriptive_name AS account_name
            FROM campaign_criterion
            WHERE
                campaign_criterion.negative = True AND
                campaign_criterion.type = 'KEYWORD' AND
                campaign.status = 'ENABLED'
        """