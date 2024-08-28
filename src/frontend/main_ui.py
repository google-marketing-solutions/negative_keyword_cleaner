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
import re
import textwrap
import time
from typing import Callable

import pandas as pd
import streamlit as st
import yaml
from langchain import prompts, chains
from streamlit_elements import elements, mui

import frontend.components.mui_components as mui_comp
from frontend import models
from frontend.components import sidebar
from utils import auth, data_helper, event_helper
from utils import llm_helper
from utils.event_helper import session_state_manager

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.Logger(__name__)

_SCHEMA_EVALUATIONS = {}

_DEBUG_SCORING_LIMIT = -1  # No limit: -1

_URL_REGEX = (
    r"^((http|https)://)[-a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\."
    r"[a-z]{2,6}\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)$"
)


def _set_manual_context(
    state_manager: session_state_manager,
) -> Callable[[], None]:
  def _set_manual_context_callback():
    state_manager.set("manual_context", True)
    pass

  return _set_manual_context_callback


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
  _handle_advertiser_information(state_manager, llm)
  _load_customers_and_campaigns(state_manager)
  _sample_and_score_keywords(state_manager)
  _score_remaining_keywords(state_manager)
  _download_results(state_manager)


def _handle_advertiser_information(state_manager, llm):
  """Handles the advertiser's information input and processes it.

  Args:
    state_manager (session_state_manager): An object managing the session
    state of the application.
    llm: The language model used.

  Returns:
    None
  """
  with st.expander(
      "1. Advertiser Information",
      expanded=state_manager.get("context_open", True),
  ):
    company_homepage_url = st.text_input(
        "Company Homepage URL",
        placeholder="https://...",
        key="company_homepage_url",
        value=state_manager.get("company_homepage_url", ""),
        disabled=st.session_state.get("manual_context", False),
    )

    if not company_homepage_url and not state_manager.get("manual_context"):
      st.info(
          "Once I have their website URL, I can directly read and "
          "understand who this customer is.",
          icon="🧑‍🎓",
      )
      st.write("If you want to give me advertiser information directly")
      st.button(
          "Add context manually",
          on_click=_set_manual_context(state_manager),
      )
      st.stop()
    else:
      _process_company_homepage(company_homepage_url, state_manager, llm)

    _get_company_and_exclude_pitches(state_manager)

  _handle_context_ready(state_manager)


def _process_company_homepage(company_homepage_url, state_manager, llm):
  """Processes the company's homepage URL to gather its text.

  Args:
    company_homepage_url (str): The URL of the company's homepage to be
    processed.
    state_manager (session_state_manager): An Object that manages the
    application's session state
    llm: The language model used.

  Returns:
      None
  """

  if company_homepage_url or not state_manager.get("manual_context"):
    if not re.match(_URL_REGEX, company_homepage_url) and not state_manager.get(
        "manual_context"
    ):
      st.error(
          "The URL is not in a valid format. Please check it and try again."
      )
      st.stop()
    else:
      with st.spinner("I'm browsing their website..."):
        homepage_docs = models.fetch_landing_page_text(company_homepage_url)
        state_manager.set("homepage_fetched", True)

      st.success("Browsing done, I've collected enough info", icon="🧑‍🎓")

    with st.spinner(
        "I'm now consolidating everything into an executive summary "
        "(this will take a minute) ..."
    ):
      if not state_manager.get("homepage_summary", None):
        homepage_summary = models.summarize_text(
            homepage_docs, llm, verbose=True
        ).strip()
        state_manager.set("homepage_summary", homepage_summary)
      else:
        homepage_summary = state_manager.get("homepage_summary")

      st.success(
          "Summarizing done but feel free to correct anything that I've "
          "written.",
          icon="🎓",
      )


def _get_company_and_exclude_pitches(state_manager):
  """Displays the company's executive summaries.

  This method displays (positive prompt) and exclusion criteria (negative
  prompt) from the user input.

  Args:
    state_manager (session_state_manager): An Object that Manages the
    session state of the application.

  Returns:
    tuple: A tuple containing:
      - company_pitch (str): The executive summary provided by the user.
      - exclude_pitch (str): The exclusion criteria provided by the user.
  """

  homepage_summary = state_manager.get("homepage_summary", "")
  company_pitch = st.text_area(
      "✅ [Positive prompt] Advertiser's executive summary",
      help="You can add campaign information below",
      placeholder="Describe what the company is selling in a few words",
      value=homepage_summary,
      height=150,
  )

  st.info(
      "Happy to know more about what you don't want to target ads for",
      icon="🧑‍🎓",
  )

  exclude_pitch = st.text_area(
      "❌ [Negative prompt] Exclude summary",
      placeholder=(
          "Describe what you don't want to target ads for (your brand,"
          " competitor's names, other non-relevant topics ...)"
      ),
      height=50,
  )

  return company_pitch, exclude_pitch


def _handle_context_ready(state_manager):
  if not state_manager.get("context_ready"):
    st.button(
        "Continue with this context",
        on_click=event_helper.handle_continue_with_context,
    )
    st.stop()
  elif state_manager.get("context_open"):
    state_manager.set("context_open", False)
    st.rerun()


def _load_customers_and_campaigns(state_manager):
  _load_customers(state_manager)
  _load_negative_keywords(state_manager)
  _filter_campaigns(state_manager)
  _handle_filters_ready(state_manager)


def _load_customers(state_manager):
  """Loads and displays a list of customers for user selection.

  Args:
    state_manager (session_state_manager): An object that manages the session
    state of the application.

  Returns:
    None
  """
  with st.expander(
      "2. Load Customers",
      expanded=state_manager.get("load_customers_open", True),
  ):
    df = data_helper.load_customers()
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
  """Loads and displays negative keywords associated with the customers.

  Args:
    state_manager (session_state_manager): An Object that manages the
      session state of the application.

  Returns:
    None
  """
  with st.expander(
      "3. Load negative keywords",
      expanded=state_manager.get("load_keywords_open", True),
  ):
    selected_customers = state_manager.get("selected_customers", [])
    df = data_helper.load_keywords(selected_customers).query('keyword != ""')
    state_manager.set("df_keywords", df)
    number_of_neg_kw = f"{len(df):.0f}"
    st.success(
        f"I've loaded {number_of_neg_kw} negative keywords from all"
        " campaigns. Filter only the relevant campaigns!",
        icon="🎓",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total negative keywords",
        f"{len(df):.0f}".replace(",", " "),
    )
    col2.metric(
        "Total unique keywords",
        f"{len(df.keyword.nunique()):.0f}".replace(",", " "),
    )
    col3.metric(
        "Total campaigns",
        f"{len(df.campaign_id.nunique()):.0f}".replace(",", " "),
    )


def _filter_campaigns(state_manager):
  """Filters negative keywords based on selected campaigns.

  Args:
    state_manager (session_state_manager): An Object that manages the
      session state of the application.

  Returns:
    None
  """

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
      filtered_keywords = filtered_keywords.query("campaign_name in @options")
      state_manager.set("filtered_keywords", filtered_keywords)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Selected negative keywords",
        f"{len(filtered_keywords):.0f}".replace(",", " "),
    )
    col2.metric(
        "Selected Unique keywords",
        f"{len(filtered_keywords.keyword.nunique()):.0f}".replace(",", " "),
    )
    col3.metric(
        "Selected campaigns",
        f"{len(filtered_keywords.campaign_id.nunique()):.0f}".replace(",", " "),
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


def _display_batch_accuracy(state_manager):
  """Calculates and displays the accuracy of evaluations.

  Args:
    state_manager: The session state manager that holds the application's
      state.
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
  formatted_facts = models.format_scoring_fragment(
      state_manager.get("evaluations") or _SCHEMA_EVALUATIONS
  )
  keywords = state_manager.get("sampled_keywords")
  formatted_keywords = json.dumps(
      [keyword.strip("'\"") for keyword in keywords["keyword"].values.tolist()],
      ensure_ascii=False,
  )
  prompt = _create_prompt()
  scoring_llm = llm_helper.select_llm(state_manager.get("config"))

  scored_keywords = state_manager.get("scored_keywords", None)
  if not scored_keywords:
    with st.spinner("Scoring a new batch of keywords..."):
      llm_chain = chains.LLMChain(prompt=prompt, llm=scoring_llm, verbose=True)
      scored_keywords = llm_chain.run({
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
          "decision_allowed_values": ", ".join(
              x.value for x in models.ScoreDecision
          ),
          "batch_size": state_manager.get("batch_size"),
      })
      state_manager.set("scored_keywords", scored_keywords)


def _create_prompt():
  template = textwrap.dedent("""\
            You are a machine learning model that analyzes negative keywords for Google Ads campaigns. For each provided negative keyword, determine if it should be KEPT (to prevent ads from showing for that search term) or REMOVED (to allow ads to show for searches containing that term).

            Provide your analysis in the following YAML format for each keyword:

            - keyword: <THE_NEGATIVE_KEYWORD>
              reason: <CONCISE_EXPLANATION_OF_THE_DECISION>
              decision: <KEEP or REMOVE>

            The 'reason' field should be a well-reasoned explanation based on these factors:

            1. Relevance: Is the keyword closely related to the advertised products/services?
            2. Search Intent: Does the keyword align with the target audience's purchase intent?
            3. Brand Protection: Does the keyword prevent ads from appearing in damaging contexts?
            4. Campaign Objectives: Does the keyword align with the overall campaign goals?

            Do not use any quotes (' or ") in the 'reason' field nor mention the initial keyword.
            Your output should strictly follow the specified YAML format, with no additional text.

            KEEP means: Keep this keyword in the Negative Keyword list to prevent ads from showing for that search term.
            REMOVE means: Remove this keyword from the Negative Keyword list to allow ads to show for searches containing that term.

            Context:
            {company_segment}

            Previous Analysis:
            {facts_segment}

            Analyze the following keywords:
            {keywords_segment}
            """)
  prompt = prompts.PromptTemplate(
      template=template,
      input_variables=[
          "company_segment",
          "facts_segment",
          "keywords_segment",
          "decision_allowed_values",
          "batch_size",
      ],
  )
  return prompt


def _display_scored_keywords(state_manager):
  scored_keywords = state_manager.get("scored_keywords")
  parsed_scored_keywords = models.parse_scoring_response(scored_keywords)
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


def _split_and_display_keywords(state_manager):
  # Retrieve necessary data from state_manager
  parsed_scored_keywords = state_manager.get("parsed_scored_keywords", [])
  keywords = state_manager.get("sampled_keywords")
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
  keyword_feedback_eval = state_manager.get("keyword_feedback_eval", None)
  if keywords_to_remove or keywords_to_keep:
    with elements("cards"):
      with mui.Grid(container=True):
        # Display keywords the LLM suggests to target (REMOVE from negatives)
        with mui.Grid(item=True, xs=True):
          mui.Typography(
              f"I think you should target ({len(keywords_to_remove)}):",
              variant="h5",
              sx={"mb": 2},
          )
          with mui.Stack(spacing=2, direction="column", useFlexGap=True):
            if not keywords_to_remove:
              mui.Typography("No more.")
            for item in keywords_to_remove:
              mui_comp.render_item_card(
                  item,
                  keyword_feedback_eval=keyword_feedback_eval,
                  df_keywords=keywords,
              )
        # Divider between the two categories
        mui.Divider(orientation="vertical", flexItem=True, sx={"mx": 4})
        # Display keywords the LLM suggests to keep as negatives
        with mui.Grid(item=True, xs=True):
          mui.Typography(
              f"I think you shouldn't target ({len(keywords_to_keep)}):",
              variant="h5",
              sx={"mb": 2},
          )
          with mui.Stack(spacing=2, direction="column", useFlexGap=True):
            if not keywords_to_keep:
              mui.Typography("No more.")
            for item in keywords_to_keep:
              mui_comp.render_item_card(
                  item,
                  keyword_feedback_eval=keyword_feedback_eval,
                  df_keywords=keywords,
              )
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
  keywords_to_score = filtered_keywords

  while True:
    random_state = state_manager.get("random_state", models.get_random_state())
    keywords = models.sample_batch(
        keywords_to_score,
        batch_size=50,
        exclude_keywords=scoring_seen_kws,
        random_state=random_state,
    )
    if not keywords:
      scoring_bar.progress(1.0, text="")
      st.markdown("Done ✅")
      break

    formatted_facts = models.format_scoring_fragment(
        state_manager.get("evaluations") or _SCHEMA_EVALUATIONS
    )

    formatted_keywords = yaml.dump(
        keywords["keyword"].tolist(),
        allow_unicode=True,
        default_flow_style=False,
    )
    prompt = _create_prompt()
    scoring_llm = llm_helper.select_llm(state_manager.get("config"))
    llm_chain = chains.LLMChain(prompt=prompt, llm=scoring_llm, verbose=True)

    latest_scored_keywords = llm_chain.run({
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
        "decision_allowed_values": ", ".join(
            x.value for x in models.ScoreDecision
        ),
        "batch_size": state_manager.get("batch_size"),
    })

    try:
      parsed_scored_keywords = models.parse_scoring_response(
          latest_scored_keywords
      )
    except yaml.scanner.ScannerError:
      continue

    scoring_kws_evals.extend(parsed_scored_keywords)
    scoring_seen_kws.update(keywords["keyword"].tolist())

    curr = len(scoring_kws_evals)
    all_kws = len(keywords_to_score)
    scoring_bar.progress(value=curr / all_kws,
                         text=scoring_progress_text + f" {curr}/{all_kws}")

  if scoring_kws_evals:
    state_manager.set("scoring_kws_evals", scoring_kws_evals)


def _download_results(state_manager):
  if state_manager.get("scoring_kws_evals", None):
    _prepare_and_display_downloads(state_manager)
  _handle_stop_training(state_manager)


def _format_customer_id(cid: int) -> str:
  """Format the customer ID into a more readable string format.

  Args:
    cid (int): The customer ID to be formatted.

  Returns:
    str: The formatted customer ID.
  """
  str_cid = str(cid)
  return f"{str_cid[:3]}-{str_cid[3:6]}-{str_cid[6:]}"


def _prepare_and_display_downloads(state_manager):
  cached_scoring_kws_evals = state_manager.get("scoring_kws_evals")
  filtered_keywords = state_manager.get("filtered_keywords")

  # Prepare dataframes for removal and keeping
  df_to_remove, df_to_keep = _prepare_dataframes(
      cached_scoring_kws_evals, filtered_keywords
  )

  # Display DataFrames and download buttons
  st.dataframe(
      df_to_remove,
      height=200,
      column_config={
          "campaign_id": st.column_config.TextColumn("campaign_id"),
          "adgroup_id": st.column_config.TextColumn("adgroup_id"),
          "account_id": st.column_config.TextColumn("account_id"),
      },
  )
  st.download_button(
      "Download keywords to remove found by Student",
      df_to_remove.to_csv(index=False),
      file_name="negative_keywords_to_remove.csv",
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
              "Negative " if df_entry.adgroup_name else "Campaign Negative "
          ) + (
              "Broad"
              if df_entry.match_type == "BROAD"
              else "Phrase"
              if df_entry.match_type == "PHRASE"
              else "Exact"
          ),
          "Status": "Removed",
          "Student Reason": student_eval.reason,
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
              "Negative " if df_entry.adgroup_name else "Campaign Negative "
          ) + (
              "Broad"
              if df_entry.match_type == "BROAD"
              else "Phrase"
              if df_entry.match_type == "PHRASE"
              else "Exact"
          ),
          "Status": "Enabled",
          "Student Reason": student_eval.reason,
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
    # Retrieve filtered_keywords and evaluations from state_manager
    filtered_keywords = state_manager.get("filtered_keywords", pd.DataFrame())
    evaluations = state_manager.get("evaluations", {})

    # Proceed only if filtered_keywords is not empty
    if not filtered_keywords.empty:
      formatted_evals = [
          {
              "keyword": kw,
              "original_keyword": filtered_keywords.loc[
                  filtered_keywords["keyword"] == kw, "original_keyword"
              ].values[0],
              "human_decision": human_eval.decision,
              "human_reason": human_eval.reason,
              "campaign_name": filtered_keywords.loc[
                  filtered_keywords["keyword"] == kw, "campaign_name"
              ].values[0],
              "campaign_id": filtered_keywords.loc[
                  filtered_keywords["keyword"] == kw, "campaign_id"
              ].values[0],
              "adgroup_id": filtered_keywords.loc[
                  filtered_keywords["keyword"] == kw, "adgroup_id"
              ].values[0],
          }
          for kw, human_eval in evaluations.items()
          if kw in filtered_keywords["keyword"].values
      ]
      df_output = pd.DataFrame(formatted_evals)

      st.dataframe(df_output, height=200)
      st.download_button(
          "Download human scorings",
          df_output.to_csv(index=False),
          file_name="negative_keywords_used_to_train_student.csv",
      )
    else:
      st.warning("No filtered keywords are available.")
