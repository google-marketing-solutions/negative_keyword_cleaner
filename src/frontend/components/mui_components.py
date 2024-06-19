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

import pandas as pd
from streamlit_elements import mui, lazy

from frontend import models
from frontend.models import KeywordEvaluation
from utils import event_helper


def render_item_card(
    item: models.KeywordEvaluation,
    state_manager: event_helper.SessionStateManager,
    keyword_feedback_eval: KeywordEvaluation = None,
    df_keywords: pd.DataFrame = None,
) -> None:
  """
  Render a card UI component for a keyword evaluation item

  Parameters:
      item (models.KeywordEvaluation): The keyword evaluation to be rendered in the card.
      state_manager: The session state manager.
      keyword_feedback_eval: Evaluation feedback, if any.
      df_keywords: List of all keywords and campaign association
  """
  kw_lines = df_keywords.loc[df_keywords.keyword == item.keyword]
  kw_campaigns = kw_lines.campaign_name.tolist()

  card_style = {"display": "flex", "flexDirection": "column", "borderRadius": 3}
  card_header_style = {"background": "rgba(250, 250, 250, 0.1)"}
  with mui.Card(key="first_item", sx=card_style, elevation=1):
    mui.CardHeader(
        title=item.keyword,
        titleTypographyProps={"variant": "h6"},
        sx=card_header_style,
    )
    with mui.CardContent(sx={"flex": 1, "pt": 0, "pb": 0}):
      with mui.Table(), mui.TableBody():
        if (
            keyword_feedback_eval is not None
            and keyword_feedback_eval.keyword == item.keyword
        ):
          create_table_row(
              "Human Reason",
              "Explain your rating (e.g. 'Competitor product')",
              is_textfield=True,
              default_value=keyword_feedback_eval.reason,
              human_reason=lambda event: event_helper.handler_human_reason(
                  event=event
              ),
          )
          create_table_row(
              "Human Decision",
              None,
              decisions=models.ScoreDecision,
              default_value=keyword_feedback_eval.decision,
              human_decision=lambda event, value: event_helper.handler_human_decision(
                  event, value
              ),
          )
        else:
          create_table_row(len(kw_campaigns), ", ".join(kw_campaigns))
          create_table_row("AI Reason", item.reason or "Empty")

    # Add actions
    create_action_section(
        item=item,
        keyword_feedback_eval=keyword_feedback_eval,
    )


def create_action_section(
    item: models.KeywordEvaluation,
    keyword_feedback_eval: models.KeywordEvaluation = None,
):
  if (
      keyword_feedback_eval is not None
      and keyword_feedback_eval.keyword == item.keyword
  ):
    # Feedback actions
    with mui.CardActions(disableSpacing=True, sx={"margin-top": "auto"}):
      mui.Button(
          "Cancel",
          onClick=event_helper.handler_cancel_human_eval(),
          sx={"margin-right": "auto", "color": "#999999"},
      )
      mui.Button(
          "Save human feedback",
          color="success",
          onClick=event_helper.define_handler_save_human_eval(
              llm_eval=item,
              keyword_feedback_eval=keyword_feedback_eval,
          ),
      )
  else:
    # Generic actions
    with mui.CardActions(disableSpacing=True, sx={"margin-top": "auto"}):
      mui.Button(
          "Disagree with Student",
          color="error",
          onClick=event_helper.define_handler_scoring(
              llm_eval=item, human_agree_with_llm=False
          ),
          sx={"margin-right": "auto"},
      )

      mui.Button(
          "Agree with Student",
          color="success",
          onClick=event_helper.define_handler_scoring(
              llm_eval=item,
              human_agree_with_llm=True,
          ),
      )


def create_table_row(
    label,
    content,
    is_textfield=False,
    default_value=None,
    human_reason=None,
    human_decision=None,
    decisions=None,
):
  table_row_style = {"&:last-child td, &:last-child th": {"border": 0}}
  with mui.TableRow(sx=table_row_style):
    with mui.TableCell(component="th", scope="row", sx={"p": 0}):
      if type(label) is int:
        avatar = mui.Avatar(children=label)
        mui.Chip(label=f"Campaign{'s' if label > 1 else ''}", avatar=avatar)
      else:
        mui.Chip(label=label)
    with mui.TableCell():
      if is_textfield:
        mui.TextField(
            multiline=True,
            placeholder=content,
            defaultValue=default_value,
            fullWidth=True,
            onChange=lazy(human_reason),
        )
      elif decisions is not None:
        with mui.Select(value=default_value, onChange=human_decision):
          for dec in decisions:
            if dec.name == "KEEP":
              mui.MenuItem("Don't target this keyword", value=dec.value)
            elif dec.name == "REMOVE":
              mui.MenuItem("Target this keyword", value=dec.value)
      else:
        mui.Typography(
            content,
            noWrap=True if (type(label) == int) else False,
            paragraph=True if label == "AI Reason" else False,
        )


def create_table(rows, key=None):
  """
  Creates a simple table.
  """
  with mui.Table(key=key):
    with mui.TableBody():
      for row in rows:
        with mui.TableRow(
            sx={"&:last-child td, &:last-child th": {"border": 0}}
        ):
          for cell in row:
            mui.TableCell()(mui.Typography(cell))
