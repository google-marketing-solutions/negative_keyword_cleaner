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
from utils.event_helper import SessionStateManager

logging.getLogger().setLevel(logging.DEBUG)
logger = logging.Logger(__name__)

_SCHEMA_EVALUATIONS = {}

_DEBUG_SCORING_LIMIT = -1  # No limit: -1

_URL_REGEX = r"^((http|https)://)[-a-zA-Z0-9@:%._\\+~#?&//=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%._\\+~#?&//=]*)$"


def _set_manual_context(
    state_manager: SessionStateManager,
) -> Callable[[], None]:
  def _set_manual_context_callback():
    state_manager.set("manual_context", True)
    pass

  return _set_manual_context_callback


def display_page(state_manager: SessionStateManager) -> None:
  st.header("Welcome to Negative Keyword Cleaner")
  st.info(
      "I am your AI student, ready to learn from your account to identify "
      "negative keywords that block relevant ads.",
      icon="🧑‍🎓",
  )

  auth.authenticate_user()
  sidebar.display_sidebar_component()

  llm = llm_helper.select_llm(state_manager.get("config"))

  ##
  # 1. Company Details
  #

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
          "Add context manually", on_click=_set_manual_context(state_manager)
      )
      st.stop()
    else:
      if len(company_homepage_url) > 0 or not state_manager.get(
          "manual_context"
      ):
        if not re.match(
            _URL_REGEX, company_homepage_url
        ) and not state_manager.get("manual_context"):
          state_manager.set("company_homepage_url", None)
          company_homepage_url = None
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
              "written. ",
              icon="🎓",
          )

      company_pitch = st.text_area(
          "✅ [Positive prompt] Advertiser's executive summary",
          help="You can add campaign information below",
          placeholder="Describe what the company is selling in a few words",
          value=homepage_summary if "homepage_summary" in locals() else "",
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
              " competitor's names, other non relevant topics ...)"
          ),
          height=50,
      )

  if not state_manager.get("context_ready"):
    st.button(
        "Continue with this context",
        on_click=event_helper.handle_continue_with_context,
    )
    st.stop()
  elif state_manager.get("context_open"):
    state_manager.set("context_open", False)
    st.rerun()

  ##
  # 2. Loads keywords
  #

  def _format_customer_id(cid: int) -> str:
    """
    Format the customer ID into a more readable string format.

    Parameters:
    cid (int): The customer ID to be formatted.

    Returns:
    str: The formatted customer ID.
    """
    str_cid = str(cid)
    return f"{str_cid[:3]}-{str_cid[3:6]}-{str_cid[6:]}"

  with st.expander(
      "2. Load Customers",
      expanded=state_manager.get("load_customers_open", True),
  ):
    df = data_helper.load_customers()
    selected_customers = st.multiselect(
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

  with st.expander(
      "3. Load negative keywords",
      expanded=state_manager.get("load_keywords_open", True),
  ):
    df = data_helper.load_keywords(selected_customers).query('keyword != ""')
    number_of_neg_kw: str = "{:.0f}".format(len(df))
    st.success(
        f"I've loaded {number_of_neg_kw} negative keywords from all campaigns."
        " Filter only the relevant campaigns!",
        icon="🎓",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total negative keywords", "{:.0f}".format(len(df)).replace(",", " ")
    )
    col2.metric(
        "Total unique keywords",
        "{:.0f}".format(df.keyword.nunique()).replace(",", " "),
    )
    col3.metric(
        "Total campaigns",
        "{:.0f}".format(df.campaign_id.nunique()).replace(",", " "),
    )

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

    df_filtered = df.copy()
    if state_manager.get("selected_campaigns", None):
      options = state_manager.get("selected_campaigns")
      df_filtered = df_filtered.query("campaign_name in @options")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Selected negative keywords",
        "{:.0f}".format(len(df_filtered)).replace(",", " "),
    )
    col2.metric(
        "Selected Unique keywords",
        "{:.0f}".format(df_filtered.keyword.nunique()).replace(",", " "),
    )
    col3.metric(
        "Selected campaigns",
        "{:.0f}".format(df_filtered.campaign_id.nunique()).replace(",", " "),
    )

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

  ##
  # 3. Samples and Scores the sampled batch.
  #
  if "evaluations" not in state_manager.list_keys():
    event_helper.reset_evaluations()
  if "eval_pairs" not in state_manager.list_keys():
    event_helper.reset_eval_pairs()

  evaluations = state_manager.get("evaluations")

  if not state_manager.get("stop_training"):
    random_state = state_manager.get("random_state", models.get_random_state())
    df_keywords = models.sample_batch(
        df_filtered,
        batch_size=state_manager.get("batch_size"),
        exclude_keywords=set(evaluations.keys()),
        random_state=random_state,
    )

    formatted_facts = models.format_scoring_fragment(
        state_manager.get("evaluations") or _SCHEMA_EVALUATIONS
    )
    # formatted_keywords = yaml.dump(df_keywords['keyword'].tolist(),
    #                           allow_unicode=True)
    formatted_keywords = json.dumps(
        df_keywords["keyword"].values.tolist(), ensure_ascii=False
    )

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

            Do not use quotes in the 'reason' field. 
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

    scoring_llm = llm_helper.select_llm(state_manager.get("config"))

    scored_keywords = state_manager.get("scored_keywords", None)
    if not scored_keywords:
      with st.spinner("Scoring a new batch of keywords..."):
        llm_chain = chains.LLMChain(
            prompt=prompt, llm=scoring_llm, verbose=True
        )
        print(
            llm_chain,
            company_pitch,
            exclude_pitch,
            formatted_facts,
            formatted_keywords,
        )

        scored_keywords = llm_chain.run({
            "company_segment": "\n\n".join(
                filter(None, [company_pitch, exclude_pitch])
            ),
            "facts_segment": formatted_facts,
            "keywords_segment": formatted_keywords,
            "decision_allowed_values": ", ".join(
                x.value for x in models.ScoreDecision
            ),
            "batch_size": state_manager.get("batch_size"),
        })
        state_manager.set("scored_keywords", scored_keywords)

    logger.error(scored_keywords)
    parsed_scored_keywords = models.parse_scoring_response(scored_keywords)
    if "batch_scored_keywords" not in state_manager.list_keys():
      state_manager.set("batch_scored_keywords", set())
    if "batch_eval_pairs" not in state_manager.list_keys():
      state_manager.set("batch_eval_pairs", list())
    scored_set = state_manager.get("batch_scored_keywords", set())
    eval_pairs = state_manager.get("batch_eval_pairs", list())

    if not evaluations:
      st.info(
          "Help me improve my knowledge by correcting my following "
          "guesses. I will get better over time.",
          icon="🎓",
      )
    else:
      st.success(
          f"I've learned from {len(evaluations)} human evaluations. "
          "Keep on correcting me to improve my accuracy!",
          icon="🎓",
      )

    # Splits keywords on the LLM decision, to simplify the review process
    keywords_to_remove = []
    keywords_to_keep = []
    keywords_unknown = []
    for item in parsed_scored_keywords:
      if item.keyword in state_manager.get("scored_set"):
        continue
      match item.decision:
        case models.ScoreDecision.KEEP:
          keywords_to_keep.append(item)
        case models.ScoreDecision.REMOVE:
          keywords_to_remove.append(item)
        case models.ScoreDecision.UNKNOWN:
          keywords_unknown.append(item)

    # Tracks keyword when a user disagrees.
    keyword_feedback_eval = state_manager.get("keyword_feedback_eval", None)

    if keywords_to_remove or keywords_to_keep:
      with elements("cards"):
        with mui.Grid(container=True):
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
                    df_keywords=df_keywords,
                    state_manager=state_manager,
                )
          mui.Divider(orientation="vertical", flexItem=True, sx={"mx": 4})
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
                    df_keywords=df_keywords,
                    state_manager=state_manager,
                )
    else:
      event_helper.score_batch_evals()

      # Computes each batch evaluation accuracy.
      epoch_eval_pairs = state_manager.get("epoch_eval_pairs", [])
      epoch_accurracies = [
          sum(p.llm_decision == p.human_decision for p in eval_pair)
          / len(eval_pair)
          for eval_pair in epoch_eval_pairs
      ]
      print("epoch_accurracies:", epoch_accurracies)

      col3, col4 = st.columns(2)
      with col3:
        if epoch_accurracies:
          if len(epoch_accurracies) > 1:
            delta_accuracy = (
                epoch_accurracies[-1] - epoch_accurracies[-2]
            ) / epoch_accurracies[-2]
            delta_accuracy = f"{delta_accuracy:.0%}"
          else:
            delta_accuracy = None
          st.metric(
              label="Accuracy (last batch)",
              value=f"{epoch_accurracies[-1]:.0%}",
              delta=delta_accuracy,
          )
      with col4:
        batch_count = len(scored_set)
        batch_size = len(parsed_scored_keywords)
        st.progress(
            batch_count / batch_size,
            text=f"Batch completion {batch_count:d}/{batch_size:d}",
        )

      with elements("placeholder"):
        mui.Typography(
            "You can fetch a new batch to improve the accuracy of the Student",
            align="center",
            sx={"mt": 2},
        )
      st.button("Sample a new batch", on_click=event_helper.handle_sample_batch)

  ##
  # 4. Run the student on the remaining keywords
  #

  count_remaining = df_filtered.keyword.nunique() - len(evaluations)
  st.header(f"Score the remaining {count_remaining:,} keywords")
  st.markdown(
      "**Will NOT modify your account**, it's purely a scoring procedure."
  )

  score_remaining = st.button("Score remaining keywords", key="score_remaining")
  if score_remaining:
    scoring_progress_text = (
        "Your fine-tuned AI-Student is now scoring the remaining keywords... "
    )
    scoring_bar = st.progress(0, text=scoring_progress_text)
    scoring_seen_kws = set(evaluations.keys())
    scoring_kws_evals = list()

    if _DEBUG_SCORING_LIMIT > 0:
      df_to_score = df_filtered.iloc[:_DEBUG_SCORING_LIMIT]
    else:
      df_to_score = df_filtered

    while True:
      random_state = state_manager.get(
          "random_state", models.get_random_state()
      )
      df_keywords = models.sample_batch(
          df_to_score,
          batch_size=50,
          exclude_keywords=scoring_seen_kws,
          random_state=random_state,
      )
      if len(df_keywords) == 0:
        scoring_bar.progress(1.0, text="")
        st.markdown("Done ✅")
        break

      formatted_facts = models.format_scoring_fragment(
          state_manager.get("evaluations") or _SCHEMA_EVALUATIONS
      )

      formatted_keywords = yaml.dump(
          df_keywords["keyword"].tolist(),
          allow_unicode=True,
          default_flow_style=False,
      )
      llm_chain = chains.LLMChain(prompt=prompt, llm=scoring_llm, verbose=True)

      latest_scored_keywords = llm_chain.run({
          "company_segment": "\n\n".join(
              filter(None, [company_pitch, exclude_pitch])
          ),
          "facts_segment": formatted_facts,
          "keywords_segment": formatted_keywords,
          "decision_allowed_values": ", ".join(
              x.value for x in models.ScoreDecision
          ),
          "batch_size": state_manager.get("batch_size"),
      })
      logger.warning(latest_scored_keywords)

      # Parses the results.
      try:
        parsed_scored_keywords = models.parse_scoring_response(
            latest_scored_keywords
        )
      except yaml.scanner.ScannerError as inst:
        # Skips this failed batch.
        logger.error(f"Failed batch with error: {inst}")
        continue

      scoring_kws_evals.extend(parsed_scored_keywords)

      # Marks them as seen.
      scoring_seen_kws.update(df_keywords["keyword"].tolist())

      # Updates the progress bar
      curr = len(scoring_kws_evals)
      N = len(df_to_score)
      scoring_bar.progress(
          curr / N, text=scoring_progress_text + f" {curr}/{N}"
      )

    # Keeps the results in cache
    if scoring_kws_evals:
      state_manager.set("scoring_kws_evals", scoring_kws_evals)

  # Displays the download button if a scoring has been run.
  if state_manager.get("scoring_kws_evals", None):
    cached_scoring_kws_evals = state_manager.get("scoring_kws_evals")
    # Prepares the keywords to remove for download.
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
        for df_entry in df_filtered.itertuples()
        if df_entry.keyword == student_eval.keyword
        and (df_entry.campaign_name or df_entry.adgroup_id)
    ]

    def get_df_values(df, keyword, columns):
      # Fetches values for specified columns based on the keyword
      return {
          col: df.loc[df["keyword"] == keyword, col].values[0]
          for col in columns
      }

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
                df_filtered,
                student_eval.keyword,
                ["original_keyword", "campaign_id", "adgroup_id", "account_id"],
            ),
        }
        for student_eval in cached_scoring_kws_evals
        if student_eval.decision == models.ScoreDecision.KEEP
        for df_entry in df_filtered.itertuples()
        if df_entry.keyword == student_eval.keyword
        and (df_entry.campaign_name or df_entry.adgroup_id)
    ]

    df_to_remove = pd.DataFrame(formatted_evals_to_remove)
    df_to_keep = pd.DataFrame(formatted_evals_to_keep)
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

  ##
  # 5. Download scored keywords
  #

  st.header("Download negative keywords to remove")
  st.markdown(
      "Identified by the AI Student **and confirmed by a human expert**!"
  )

  stop_training = st.button("Stop the training", key="stop_training")
  if stop_training:
    keyword_scored = []
    formatted_evals = [
        {
            "keyword": kw,
            "original_keyword": df_filtered.loc[
                df_filtered["keyword"] == kw, "original_keyword"
            ].values[0],
            "human_decision": human_eval.decision,
            "human_reason": human_eval.reason,
            "campaign_name": df_filtered.loc[
                df_filtered["keyword"] == kw, "campaign_name"
            ].values[0],
            "campaign_id": df_filtered.loc[
                df_filtered["keyword"] == kw, "campaign_id"
            ].values[0],
            "adgroup_id": df_filtered.loc[
                df_filtered["keyword"] == kw, "adgroup_id"
            ].values[0],
        }
        for kw, human_eval in evaluations.items()
        if kw in df_filtered["keyword"].values
    ]
    df_output = pd.DataFrame(formatted_evals)

    st.dataframe(
        df_output,
        height=200,
        column_config={
            "campaign_id": st.column_config.TextColumn("campaign_id"),
            "adgroup_id": st.column_config.TextColumn("adgroup_id"),
        },
    )

    st.download_button(
        "Download human scorings",
        df_output.to_csv(index=False),
        file_name="negative_keywords_used_to_train_student.csv",
    )
