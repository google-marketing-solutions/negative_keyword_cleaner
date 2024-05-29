import collections
import unittest

import models

EXPECTED_RESPONSE = """
- keyword: keyword1
  reason: Reason for keeping keyword1
  decision: KEEP
- keyword: keyword2
  reason: Reason for removing keyword2
  decision: REMOVE
"""


class ModelsTest(unittest.TestCase):

  def test_format_scoring_fragment(self):
    evaluations = collections.OrderedDict([
        (
            "keyword1",
            models.KeywordEvaluation(
                keyword="keyword1",
                decision=models.ScoreDecision.KEEP,
                reason="Reason for keeping keyword1",
            ),
        ),
        (
            "keyword2",
            models.KeywordEvaluation(
                keyword="keyword2",
                decision=models.ScoreDecision.REMOVE,
                reason="Reason for removing keyword2",
            ),
        ),
    ])
    expected_fragment = EXPECTED_RESPONSE
    self.assertEqual(
        models.format_scoring_fragment(evaluations), expected_fragment
    )

  def test_parse_scoring_response(self):
    expected_evaluations = [
        models.KeywordEvaluation(
            keyword="keyword1",
            decision=models.ScoreDecision.KEEP,
            reason="Reason for keeping keyword1",
        ),
        models.KeywordEvaluation(
            keyword="keyword2",
            decision=models.ScoreDecision.REMOVE,
            reason="Reason for removing keyword2",
        ),
    ]
    self.assertEqual(
        models.parse_scoring_response(EXPECTED_RESPONSE), expected_evaluations
    )


if __name__ == "__main__":
  unittest.main()
