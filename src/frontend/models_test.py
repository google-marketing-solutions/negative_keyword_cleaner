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

import collections
import unittest

from frontend import models
import pandas as pd

# format_scoring_fragment rewrites "reason:" to "reason::" by design.
EXPECTED_FRAGMENT = (
    "- keyword: keyword1\n"
    "  reason:: Reason for keeping keyword1\n"
    "  decision: KEEP\n"
    "  priority_score: 4\n"
    "- keyword: keyword2\n"
    "  reason:: Reason for removing keyword2\n"
    "  decision: REMOVE\n"
    "  priority_score: 1\n"
)


class ModelsTest(unittest.TestCase):

  def test_format_scoring_fragment(self):
    evaluations = collections.OrderedDict([
        (
            "keyword1",
            models.KeywordEvaluation(
                keyword="keyword1",
                decision=models.ScoreDecision.KEEP,
                reason="Reason for keeping keyword1",
                priority_score=4,
            ),
        ),
        (
            "keyword2",
            models.KeywordEvaluation(
                keyword="keyword2",
                decision=models.ScoreDecision.REMOVE,
                reason="Reason for removing keyword2",
                priority_score=1,
            ),
        ),
    ])
    self.assertEqual(
        models.format_scoring_fragment(evaluations), EXPECTED_FRAGMENT
    )

  def test_keyword_evaluation_model_normalizes_decision(self):
    parsed = models.KeywordEvaluations(
        evaluations=[
            models.KeywordEvaluationModel(
                keyword="keyword1",
                reason="Reason for keeping keyword1",
                decision="keep",
                priority_score=4,
            ),
            models.KeywordEvaluationModel(
                keyword="keyword2",
                reason="Reason for removing keyword2",
                decision="REMOVE",
                priority_score=5,
            ),
        ]
    )
    self.assertEqual(
        parsed.evaluations[0].decision, models.ScoreDecision.KEEP
    )
    self.assertEqual(parsed.evaluations[0].priority_score, 4)
    self.assertEqual(
        parsed.evaluations[1].decision, models.ScoreDecision.REMOVE
    )
    self.assertEqual(parsed.evaluations[1].priority_score, 5)

  def test_prepare_dataframes_preserves_priority_and_columns(self):
    filtered_keywords = pd.DataFrame([
        {
            "keyword": "kw_remove",
            "original_keyword": "[kw_remove]",
            "match_type": "EXACT",
            "campaign_name": "Camp A",
            "campaign_id": "111",
            "adgroup_id": "222",
            "adgroup_name": "AdGroup 1",
            "account_id": "333",
            "account_name": "Account 1",
        },
        {
            "keyword": "kw_keep",
            "original_keyword": '"kw_keep"',
            "match_type": "PHRASE",
            "campaign_name": "Camp B",
            "campaign_id": "444",
            "adgroup_id": "",
            "adgroup_name": "",
            "account_id": "333",
            "account_name": "Account 1",
        },
    ])
    evals = [
        models.KeywordEvaluation(
            keyword="kw_remove",
            decision=models.ScoreDecision.REMOVE,
            reason="Blocks relevant queries",
            priority_score=5,
        ),
        models.KeywordEvaluation(
            keyword="kw_keep",
            decision=models.ScoreDecision.KEEP,
            reason="Protects brand",
            priority_score=3,
        ),
    ]
    df_to_remove, df_to_keep = models.prepare_dataframes(
        evals, filtered_keywords
    )
    self.assertEqual(len(df_to_remove), 1)
    self.assertEqual(df_to_remove.iloc[0]["Keyword"], "kw_remove")
    self.assertEqual(df_to_remove.iloc[0]["Priority Score"], 5)
    self.assertEqual(df_to_remove.iloc[0]["Criterion Type"], "Negative Exact")

    self.assertEqual(len(df_to_keep), 1)
    self.assertEqual(df_to_keep.iloc[0]["Keyword"], "kw_keep")
    self.assertEqual(df_to_keep.iloc[0]["Priority Score"], 3)
    self.assertEqual(
        df_to_keep.iloc[0]["Criterion Type"], "Campaign Negative Phrase"
    )
    self.assertEqual(df_to_keep.iloc[0]["original_keyword"], '"kw_keep"')

    human_df = models.prepare_human_dataframe(
        filtered_keywords,
        {"kw_remove": evals[0]},
    )
    self.assertEqual(len(human_df), 1)
    self.assertEqual(human_df.iloc[0]["human_decision"], "REMOVE")
    self.assertEqual(human_df.iloc[0]["original_keyword"], "[kw_remove]")


if __name__ == "__main__":
  unittest.main()
