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
  """
  Enum class to represent different levels at which keywords can be applied in an advertising context.

  Attributes:
      ACCOUNT (str): Represents the account level.
      CAMPAIGN (str): Represents the campaign level.
      ADGROUP (str): Represents the ad group level.
  """

  ACCOUNT = "Account"
  CAMPAIGN = "Campaign"
  ADGROUP = "Adgroup"


class CustomerNames(BaseQuery):
  """
  Class for querying customer names.

  This class extends BaseQuery to execute a SQL query that selects the ID and descriptive name
  of customers who are not managers and have an enabled status both at the customer and the client level.
  """

  def __init__(self) -> None:
    self.query_text = f"""
            SELECT
                customer_client.id,
                customer_client.descriptive_name
            FROM customer_client
            WHERE
                customer_client.manager = FALSE AND
                customer_client.status = 'ENABLED' AND
                customer.status = 'ENABLED'
        """


class AdgroupNegativeKeywords(BaseQuery):
  """
  Class for querying ad group negative keywords.

  This class extends BaseQuery to execute a SQL query that selects details of negative keywords
  at the ad group level, including the criterion ID, keyword text, match type, and other relevant information.
  """

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
                '' AS adgroup_id,
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


class AccountNegativeKeywords(BaseQuery):
  """
  Class for querying account level negative keywords.

  This class extends BaseQuery to execute a SQL query that selects details of negative keywords
  at the account level, including shared set ID, keyword text, match type, and other related information.
  """

  def __init__(self) -> None:
    self.query_text = f"""
            SELECT
                shared_set.id AS criterion_id,
                'True' AS is_negative,
                shared_criterion.keyword.text AS keyword,
                shared_criterion.keyword.match_type AS match_type,
                '{KeywordLevel.ACCOUNT.value}' AS level,
                0 AS adgroup_id,
                '' AS adgroup_name,
                0 AS campaign_id,
                shared_set.name AS campaign_name,
                0 AS account_id,
                '' AS account_name,
                shared_set.type,
                shared_set.resource_name AS resource_name,
            FROM
                shared_criterion
            WHERE
                shared_set.type = NEGATIVE_KEYWORDS
        """


class CampaignsForSharedSets(BaseQuery):
  """
  Class for querying campaigns that are associated with specific shared sets.

  This class extends BaseQuery to execute a SQL query that selects details of campaigns
  linked to given shared sets, which are identified by their resource names.
  """

  def __init__(self, campaign_resource_names) -> None:
    self.query_text = f"""
            SELECT
              campaign.id,
              campaign.name,
              campaign_shared_set.shared_set
            FROM
              campaign_shared_set
            WHERE
              campaign_shared_set.shared_set IN ({','.join([f"'{name}'" for name in campaign_resource_names])})
        """
