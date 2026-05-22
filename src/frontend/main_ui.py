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

"""This module is responsible for handling the user interface of the app.

This module also handles the logic related to managing and filtering Google
Ads negative keywords.
It provides a Streamlit-based interactive UI for processing advertiser
information, scoring negative keywords, and allowing users to download the
results.
"""

import json
import logging
import textwrap

from frontend import models
from frontend.components import sidebar
from frontend.components.advertiser_info import handle_advertiser_information
from frontend.components.keyword_selector import load_customers_and_campaigns
from langchain_core.prompts import PromptTemplate
import pandas as pd
import streamlit as st
from utils import auth
from utils import event_helper
from utils import llm_helper
from utils.event_helper import session_state_manager
import yaml

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

_SCHEMA_EVALUATIONS = {}

_DEBUG_SCORING_LIMIT = -1  # No limit: -1


def display_page(state_manager: session_state_manager) -> None:
  """Displays the main UI page for the NKC application using Streamlit.

    This method sets up the interface for the user to interact with the
    application. It begins by displaying a welcome header and an
    information box about the AI  student's purpose.  The user
    authentication is handled via the `auth.authenticate_user` function.
    After authentication, it displays a sidebar for user interaction and
    calls several helper functions to manage the following tasks:

  Args:
      state_manager (session_state_manager): An object managing the session
        state of the application.

  Returns:
    None
  """

  st.header("Welcome to Negative Keyword Cleaner")
  st.info(
      "I am your AI student, ready to learn from your account to identify "
      "negative keywords that block relevant ads.",
      icon="🧑‍🎓",
  )

  auth.authenticate_user()
  sidebar.display_sidebar_component()

  llm = llm_helper.select_llm(state_manager.get("config"))

  # Call helper functions
  handle_advertiser_information(state_manager, llm)
  load_customers_and_campaigns(state_manager)
  _sample_and_score_keywords(state_manager)
  _score_remaining_keywords(state_manager)
  _download_results(state_manager)


def _display_batch_accuracy(state_manager):
  """Calculates and displays the accuracy of evaluations.

  Args:
    state_manager: The session state manager that holds the application's state.
  """
  # Retrieve epoch evaluation pairs from state_manager
  epoch_eval_pairs = state_manager.get("epoch_eval_pairs", [])
  if not epoch_eval_pairs:
    st.info("No evaluations available to display accuracy.")
    return  # No evaluations to display

  # Calculate accuracies for each batch
  epoch_accuracies = [
      sum(p.llm_decision == p.human_decision for p in eval_pair)
      / len(eval_pair)
      for eval_pair in epoch_eval_pairs
  ]
  print("epoch_accuracies:", epoch_accuracies)

  # Display the accuracy metric and progress
  col1, col2 = st.columns(2)
  with col1:
    if epoch_accuracies:
      if len(epoch_accuracies) > 1 and epoch_accuracies[-2] != 0:
        delta_accuracy = (
            epoch_accuracies[-1] - epoch_accuracies[-2]
        ) / epoch_accuracies[-2]
        delta_accuracy = f"{delta_accuracy:.0%}"
      else:
        delta_accuracy = None
      st.metric(
          label="Accuracy",
          value=f"{epoch_accuracies[-1]:.0%}",
          delta=delta_accuracy,
      )
  with col2:
    # Retrieve batch counts
    batch_scored_keywords = state_manager.get("scored_set", set())
    batch_count = len(batch_scored_keywords)
    batch_size = state_manager.get("batch_size")
    if batch_size > 0:
      progress = batch_count / batch_size
    else:
      progress = 0
    st.progress(
        progress,
        text=f"Batch completion {batch_count}/{batch_size}",
    )


def _sample_and_score_keywords(state_manager):
  # Reset batch_completed at the start of a new batch
  state_manager.set("batch_completed", False)

  # Sampling keywords
  _sample_keywords(state_manager)

  # Scoring keywords
  _score_keywords(state_manager)

  # Display thebatch accuracy
  _display_batch_accuracy(state_manager)

  # Display the scored keywords for user feedback
  _display_scored_keywords(state_manager)

  state_manager.set("batch_completed", True)

  # Provide an option to sample a new batch
  st.button("Sample a new batch", on_click=event_helper.handle_sample_batch)


def _sample_keywords(state_manager):
  random_state = state_manager.get("random_state", models.get_random_state())
  filtered_keywords = state_manager.get("filtered_keywords", pd.DataFrame())
  evaluations = state_manager.get("evaluations", {})
  keywords = models.sample_batch(
      filtered_keywords,
      batch_size=state_manager.get("batch_size"),
      exclude_keywords=set(evaluations.keys()),
      random_state=random_state,
  )
  state_manager.set("sampled_keywords", keywords)


def _score_keywords(state_manager):
  if state_manager.get("scored_keywords", None):
    return

  formatted_facts = models.format_scoring_fragment(
      state_manager.get("evaluations") or _SCHEMA_EVALUATIONS
  )
  keywords = state_manager.get("sampled_keywords")
  positive_keywords = st.session_state.get("positive_keywords", pd.DataFrame())

  pre_scored_keywords = []
  keywords_to_score = []

  for _, row in keywords.iterrows():
    is_positive_in_other_adgroup = False
    if not positive_keywords.empty:
      conflicting_positives = positive_keywords[
          (positive_keywords["keyword"] == row["keyword"])
          & (positive_keywords["campaign_id"] == row["campaign_id"])
          & (positive_keywords["adgroup_id"] != row["adgroup_id"])
      ]
      if not conflicting_positives.empty:
        is_positive_in_other_adgroup = True

    if is_positive_in_other_adgroup:
      pre_scored_keywords.append(
          models.KeywordEvaluation(
              keyword=row["keyword"],
              decision=models.ScoreDecision.KEEP,
              reason=(
                  "This keyword is positive in another adgroup in the same "
                  "campaign."
              ),
          )
      )
    else:
      keywords_to_score.append(row["keyword"])

  with st.spinner("Scoring a new batch of keywords..."):
    if keywords_to_score:
      formatted_keywords = json.dumps(
          [keyword.strip("'\"") for keyword in keywords_to_score],
          ensure_ascii=False,
      )
      prompt = _create_prompt()
      scoring_llm = llm_helper.select_llm(state_manager.get("config"))
      structured_llm = scoring_llm.with_structured_output(
          models.KeywordEvaluations
      )

      prompt_val = prompt.format(
          company_segment="\n\n".join(
              filter(
                  None,
                  [
                      state_manager.get("company_pitch", ""),
                      state_manager.get("exclude_pitch", ""),
                  ],
              )
          ),
          facts_segment=formatted_facts,
          keywords_segment=formatted_keywords,
      )

      llm_response = structured_llm.invoke(prompt_val)

      llm_scored_keywords = []
      for item in llm_response.evaluations:
        llm_scored_keywords.append(
            models.KeywordEvaluation(
                keyword=item.keyword,
                decision=models.ScoreDecision(item.decision),
                reason=item.reason,
            )
        )
    else:
      llm_scored_keywords = []

    all_scored_keywords = pre_scored_keywords + llm_scored_keywords
    state_manager.set("scored_keywords", all_scored_keywords)


def _create_prompt():
  template = textwrap.dedent("""\
            You are a machine learning model that analyzes negative keywords for Google Ads campaigns. For each provided negative keyword, determine if it should be KEPT (to prevent ads from showing for that search term) or REMOVED (to allow ads to show for searches containing that term).

            The 'reason' field should be a well-reasoned explanation based on these factors:

            1. Relevance: Is the keyword closely related to the advertised products/services?
            2. Search Intent: Does the keyword align with the target audience's purchase intent?
            3. Brand Protection: Does the keyword prevent ads from appearing in damaging contexts?
            4. Campaign Objectives: Does the keyword align with the overall campaign goals?

            Do not use any quotes (' or ") in the 'reason' field nor mention the initial keyword.

            KEEP means: Keep this keyword in the Negative Keyword list to prevent ads from showing for that search term.
            REMOVE means: Remove this keyword from the Negative Keyword list to allow ads to show for searches containing that term.

            Context:
            {company_segment}

            Previous Analysis:
            {facts_segment}

            Analyze the following keywords:
            {keywords_segment}
            """)
  prompt = PromptTemplate(
      template=template,
      input_variables=[
          "company_segment",
          "facts_segment",
          "keywords_segment",
      ],
  )
  return prompt


def _display_scored_keywords(state_manager):
  scored_keywords = state_manager.get("scored_keywords")
  parsed_scored_keywords = scored_keywords
  state_manager.set("parsed_scored_keywords", parsed_scored_keywords)

  if not state_manager.get("evaluations"):
    st.info(
        "Help me improve my knowledge by correcting my following "
        "guesses. I will get better over time.",
        icon="🎓",
    )
  else:
    st.success(
        f"I've learned from {len(state_manager.get('evaluations'))} human"
        " evaluations. Keep on correcting me to improve my accuracy!",
        icon="🎓",
    )

  _split_and_display_keywords(state_manager)


def _render_standard_card(
    item: models.KeywordEvaluation,
    df_keywords: pd.DataFrame,
    state_manager,
):
  """Renders a standard Streamlit card for a keyword evaluation."""
  kw_lines = df_keywords.loc[df_keywords.keyword == item.keyword]
  kw_campaigns = kw_lines.campaign_name.tolist()

  with st.container(border=True):
    st.markdown(f"### 🏷️ {item.keyword}")
    st.markdown(f"📌 **Campaigns:** {', '.join(kw_campaigns)}")
    st.markdown(f"🤖 **AI Reason:** _{item.reason or 'Empty'}_")
    st.divider()

    keyword_feedback_eval = state_manager.get("keyword_feedback_eval", None)

    if keyword_feedback_eval and keyword_feedback_eval.keyword == item.keyword:
      new_reason = st.text_input(
          "Explain your rating:",
          value=keyword_feedback_eval.reason,
          key=f"reason_{item.keyword}",
      )

      decision_options = [
          d.value
          for d in models.ScoreDecision
          if d != models.ScoreDecision.UNKNOWN
      ]
      current_index = (
          decision_options.index(keyword_feedback_eval.decision.value)
          if keyword_feedback_eval.decision.value in decision_options
          else 0
      )

      new_decision_str = st.selectbox(
          "Decision",
          options=decision_options,
          index=current_index,
          key=f"decision_{item.keyword}",
      )

      keyword_feedback_eval.reason = new_reason
      keyword_feedback_eval.decision = models.ScoreDecision(new_decision_str)

      c1, c2, _ = st.columns([1, 1, 2])
      with c1:
        if st.button("💾 Save", key=f"save_{item.keyword}"):
          event_helper.define_handler_save_human_eval(
              llm_eval=item,
              keyword_feedback_eval=keyword_feedback_eval,
          )()
      with c2:
        if st.button("🚫 Cancel", key=f"cancel_{item.keyword}"):
          event_helper.handler_cancel_human_eval()()
    else:
      c1, c2, _ = st.columns([1, 1, 2])
      with c1:
        if st.button("✅ Agree", key=f"agree_{item.keyword}"):
          event_helper.define_handler_scoring(
              llm_eval=item, human_agree_with_llm=True
          )()
      with c2:
        if st.button("❌ Disagree", key=f"disagree_{item.keyword}"):
          event_helper.define_handler_scoring(
              llm_eval=item, human_agree_with_llm=False
          )()


def _split_and_display_keywords(state_manager):
  # Retrieve necessary data from state_manager
  parsed_scored_keywords = state_manager.get("parsed_scored_keywords", [])
  keywords = state_manager.get("filtered_keywords")
  scored_set = state_manager.get("scored_set", set())

  # Initialize lists to categorize keywords
  keywords_to_remove = []
  keywords_to_keep = []
  keywords_unknown = []

  # Split keywords based on the LLM's decision
  for item in parsed_scored_keywords:
    if item.keyword in scored_set:
      continue

    if item.decision == models.ScoreDecision.KEEP:
      keywords_to_keep.append(item)
    elif item.decision == models.ScoreDecision.REMOVE:
      keywords_to_remove.append(item)
    else:
      keywords_unknown.append(item)

  # Track keyword feedback if a user disagrees
  if keywords_to_remove or keywords_to_keep:
    col1, col2 = st.columns(2)
    with col1:
      st.subheader(f"I think you should target ({len(keywords_to_remove)}):")
      if not keywords_to_remove:
        st.write("No more.")
      for item in keywords_to_remove:
        _render_standard_card(item, keywords, state_manager)

    with col2:
      st.subheader(f"I think you shouldn't target ({len(keywords_to_keep)}):")
      if not keywords_to_keep:
        st.write("No more.")
      for item in keywords_to_keep:
        _render_standard_card(item, keywords, state_manager)
  else:
    # If no keywords to display, proceed to score batch evaluations
    event_helper.score_batch_evals()
    state_manager.set("batch_completed", True)


def _score_remaining_keywords(state_manager):
  count_remaining = state_manager.get(
      "filtered_keywords", {}
  ).keyword.nunique() - len(state_manager.get("evaluations", {}))
  st.header(f"Score the remaining {count_remaining:,} keywords")
  st.markdown(
      "**Will NOT modify your account**, it's purely a scoring procedure."
  )

  if st.button("Score remaining keywords", key="score_remaining"):
    _process_remaining_keywords(state_manager)


def _process_remaining_keywords(state_manager):
  scoring_progress_text = (
      "Your fine-tuned AI-Student is now scoring the remaining keywords... "
  )
  scoring_bar = st.progress(0, text=scoring_progress_text)
  scoring_seen_kws = set(state_manager.get("evaluations").keys())
  scoring_kws_evals = list()

  filtered_keywords = state_manager.get("filtered_keywords")
  positive_keywords = st.session_state.get("positive_keywords", pd.DataFrame())
  keywords_to_score = filtered_keywords

  while True:
    random_state = state_manager.get("random_state", models.get_random_state())
    keywords_batch = models.sample_batch(
        keywords_to_score,
        batch_size=50,
        exclude_keywords=scoring_seen_kws,
        random_state=random_state,
    )
    if keywords_batch.empty:
      scoring_bar.progress(1.0, text="")
      st.markdown("Done ✅")
      break

    pre_scored_keywords = []
    keywords_to_llm = []

    for _, row in keywords_batch.iterrows():
      is_positive_in_other_adgroup = False
      if not positive_keywords.empty:
        conflicting_positives = positive_keywords[
            (positive_keywords["keyword"] == row["keyword"])
            & (positive_keywords["campaign_id"] == row["campaign_id"])
            & (positive_keywords["adgroup_id"] != row["adgroup_id"])
        ]
        if not conflicting_positives.empty:
          is_positive_in_other_adgroup = True

      if is_positive_in_other_adgroup:
        pre_scored_keywords.append(
            models.KeywordEvaluation(
                keyword=row["keyword"],
                decision=models.ScoreDecision.KEEP,
                reason=(
                    "This keyword is positive in another adgroup in the "
                    "same campaign."
                ),
            )
        )
      else:
        keywords_to_llm.append(row["keyword"])

    formatted_facts = models.format_scoring_fragment(
        state_manager.get("evaluations") or _SCHEMA_EVALUATIONS
    )

    formatted_keywords = yaml.dump(
        keywords_to_llm,
        allow_unicode=True,
        default_flow_style=False,
    )
    prompt = _create_prompt()
    scoring_llm = llm_helper.select_llm(state_manager.get("config"))
    chain = prompt | scoring_llm.with_structured_output(
        models.KeywordEvaluations
    )

    scored_keywords_obj = chain.invoke({
        "company_segment": "\n\n".join(
            filter(
                None,
                [
                    state_manager.get("company_pitch", ""),
                    state_manager.get("exclude_pitch", ""),
                ],
            )
        ),
        "facts_segment": formatted_facts,
        "keywords_segment": formatted_keywords,
    })

    if not scored_keywords_obj:
      continue

    llm_scored_keywords = [
        models.KeywordEvaluation(
            keyword=item.keyword,
            decision=models.ScoreDecision(item.decision.upper()),
            reason=item.reason,
        )
        for item in scored_keywords_obj.evaluations
    ]

    all_scored_keywords = pre_scored_keywords + llm_scored_keywords
    scoring_kws_evals.extend(all_scored_keywords)
    scoring_seen_kws.update(keywords_batch["keyword"].tolist())

    curr = len(scoring_kws_evals)
    all_kws = len(keywords_to_score)
    scoring_bar.progress(
        value=curr / all_kws, text=scoring_progress_text + f" {curr}/{all_kws}"
    )

  if scoring_kws_evals:
    state_manager.set("scoring_kws_evals", scoring_kws_evals)


def _download_results(state_manager):
  if state_manager.get("scoring_kws_evals", None):
    _prepare_and_display_downloads(state_manager)
  _handle_stop_training(state_manager)
  _display_consolidated_download()


def _prepare_and_display_downloads(state_manager):
  cached_scoring_kws_evals = state_manager.get("scoring_kws_evals")
  filtered_keywords = state_manager.get("filtered_keywords")

  # Prepare dataframes for removal and keeping
  df_to_remove, df_to_keep = _prepare_dataframes(
      cached_scoring_kws_evals, filtered_keywords
  )
  st.session_state["df_to_remove_student"] = df_to_remove

  # Filter for Campaign/AdGroup negatives
  df_campaign_adgroup = df_to_remove[df_to_remove["campaign_id"].ne("")]

  # Filter for Shared Lists negatives
  df_shared_lists = df_to_remove[df_to_remove["campaign_id"].eq("")]

  # Format Shared Lists for Editor
  df_shared_lists_editor = df_shared_lists[
      ["Campaign", "Keyword", "match_type"]
  ].rename(
      columns={"Campaign": "Negative Keyword List", "match_type": "Match Type"}
  )

  st.subheader("Campaign & AdGroup Level Negatives")
  st.dataframe(
      df_campaign_adgroup,
      height=200,
      column_config={
          "campaign_id": st.column_config.TextColumn("campaign_id"),
          "adgroup_id": st.column_config.TextColumn("adgroup_id"),
          "account_id": st.column_config.TextColumn("account_id"),
      },
  )
  st.download_button(
      "Download Campaign/AdGroup keywords to remove",
      df_campaign_adgroup.to_csv(index=False),
      file_name="negative_keywords_to_remove.csv",
  )

  if not df_shared_lists_editor.empty:
    st.subheader("Shared Negative Keyword Lists")
    st.dataframe(df_shared_lists_editor, height=200)
    st.download_button(
        "Download Shared Keyword Lists to remove (Editor format)",
        df_shared_lists_editor.to_csv(index=False),
        file_name="shared_negative_keywords_to_remove.csv",
    )
  st.dataframe(
      df_to_keep,
      height=200,
      column_config={
          "campaign_id": st.column_config.TextColumn("campaign_id"),
          "adgroup_id": st.column_config.TextColumn("adgroup_id"),
          "account_id": st.column_config.TextColumn("account_id"),
      },
  )
  st.download_button(
      "Download keywords to keep with reason written by Student",
      df_to_keep.to_csv(index=False),
      file_name="negative_keywords_to_keep.csv",
  )


def _prepare_human_dataframe(state_manager):
  """Prepares a DataFrame of human evaluations.

  Args:
    state_manager: The session state manager holding the application's state.

  Returns:
    A pandas DataFrame containing the human evaluations, or an empty DataFrame
    if no filtered keywords are available.
  """
  # Retrieve filtered_keywords and evaluations from state_manager
  filtered_keywords = state_manager.get("filtered_keywords", pd.DataFrame())
  evaluations = state_manager.get("evaluations", {})

  # Proceed only if filtered_keywords is not empty
  if not filtered_keywords.empty:
    formatted_evals = [
        {
            "keyword": kw,
            "original_keyword": (
                filtered_keywords.loc[
                    filtered_keywords["keyword"] == kw, "original_keyword"
                ].values[0]
            ),
            "human_decision": str(human_eval.decision),
            "human_reason": human_eval.reason,
            "campaign_name": (
                filtered_keywords.loc[
                    filtered_keywords["keyword"] == kw, "campaign_name"
                ].values[0]
            ),
            "campaign_id": (
                filtered_keywords.loc[
                    filtered_keywords["keyword"] == kw, "campaign_id"
                ].values[0]
            ),
            "adgroup_id": (
                filtered_keywords.loc[
                    filtered_keywords["keyword"] == kw, "adgroup_id"
                ].values[0]
            ),
            "Source": "Human",
        }
        for kw, human_eval in evaluations.items()
        if kw in filtered_keywords["keyword"].values
    ]
    return pd.DataFrame(formatted_evals)
  return pd.DataFrame()


def _display_consolidated_download():
  st.header("Consolidated Download")
  df_student = st.session_state.get("df_to_remove_student", pd.DataFrame())
  df_human = st.session_state.get("df_to_remove_human", pd.DataFrame())

  if not df_student.empty or not df_human.empty:
    df_consolidated = pd.concat([df_student, df_human], ignore_index=True)
    st.dataframe(df_consolidated, height=200)
    st.download_button(
        "Download consolidated keywords to remove",
        df_consolidated.to_csv(index=False),
        file_name="consolidated_negative_keywords_to_remove.csv",
    )


def _prepare_dataframes(cached_scoring_kws_evals, filtered_keywords):
  """Prepares two DataFrames.

    - df_to_remove: Keywords that the student suggests to remove from the
  negative keyword list.
    - df_to_keep: Keywords that the student suggests to keep in the negative
  keyword list.

  Args:
    cached_scoring_kws_evals: List of student evaluations (student_eval
      objects).
    filtered_keywords: DataFrame containing the filtered keywords.

  Returns:
    df_to_remove: DataFrame of keywords to remove.
    df_to_keep: DataFrame of keywords to keep.
  """

  # Helper function to fetch values from the DataFrame based on the keyword
  def get_df_values(df, keyword, columns):
    # Fetches values for specified columns based on the keyword
    return {
        col: df.loc[df["keyword"] == keyword, col].values[0] for col in columns
    }

  # Prepare the list of dictionaries for keywords to remove
  formatted_evals_to_remove = [
      {
          "Account": df_entry.account_name,
          "Campaign": df_entry.campaign_name,
          "Ad Group": df_entry.adgroup_name,
          "Keyword": student_eval.keyword,
          "Criterion Type": (
              ("Negative " if df_entry.adgroup_name else "Campaign Negative ")
              + (
                  "Broad"
                  if df_entry.match_type == "BROAD"
                  else "Phrase"
                  if df_entry.match_type == "PHRASE"
                  else "Exact"
              )
          ),
          "Status": "Removed",
          "Student Reason": student_eval.reason,
          "Source": "Student",
          "campaign_id": df_entry.campaign_id,
          "adgroup_id": df_entry.adgroup_id,
          "match_type": df_entry.match_type,
      }
      for student_eval in cached_scoring_kws_evals
      if student_eval.decision == models.ScoreDecision.REMOVE
      for df_entry in filtered_keywords.itertuples()
      if df_entry.keyword == student_eval.keyword
      and (df_entry.campaign_name or df_entry.adgroup_id)
  ]

  # Prepare the list of dictionaries for keywords to keep
  formatted_evals_to_keep = [
      {
          "Account": df_entry.account_name,
          "Campaign": df_entry.campaign_name,
          "Ad Group": df_entry.adgroup_name,
          "Keyword": student_eval.keyword,
          "Criterion Type": (
              ("Negative " if df_entry.adgroup_name else "Campaign Negative ")
              + (
                  "Broad"
                  if df_entry.match_type == "BROAD"
                  else "Phrase"
                  if df_entry.match_type == "PHRASE"
                  else "Exact"
              )
          ),
          "Status": "Enabled",
          "Student Reason": student_eval.reason,
          "Source": "Student",
          **get_df_values(
              filtered_keywords,
              student_eval.keyword,
              ["original_keyword", "campaign_id", "adgroup_id", "account_id"],
          ),
      }
      for student_eval in cached_scoring_kws_evals
      if student_eval.decision == models.ScoreDecision.KEEP
      for df_entry in filtered_keywords.itertuples()
      if df_entry.keyword == student_eval.keyword
      and (df_entry.campaign_name or df_entry.adgroup_id)
  ]

  # Convert the lists of dictionaries into DataFrames
  df_to_remove = pd.DataFrame(formatted_evals_to_remove)
  df_to_keep = pd.DataFrame(formatted_evals_to_keep)

  return df_to_remove, df_to_keep


def _handle_stop_training(state_manager):
  st.header("Download negative keywords to remove")
  st.markdown(
      "Identified by the AI Student **and confirmed by a human expert**!"
  )

  if st.button("Stop the training", key="stop_training"):
    df_output = _prepare_human_dataframe(state_manager)
    if not df_output.empty:
      df_to_remove_human = df_output[df_output["human_decision"] == "REMOVE"]
      st.session_state["df_to_remove_human"] = df_to_remove_human
      st.dataframe(df_output, height=200)
      st.download_button(
          "Download human scorings",
          df_output.to_csv(index=False),
          file_name="negative_keywords_used_to_train_student.csv",
      )
    else:
      st.warning("No filtered keywords are available.")
