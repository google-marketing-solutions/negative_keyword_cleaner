from collections import OrderedDict
from typing import Callable

import streamlit as st

from frontend import models


class SessionStateManager:
  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super(SessionStateManager, cls).__new__(cls)
    return cls._instance

  def __init__(self):
    self.initialize("scored_set", set())

  def initialize(self, key, default_value):
    """Initializes a session state variable if it does not exist."""
    if key not in st.session_state:
      st.session_state[key] = default_value

  def get(self, key, default=None):
    """Safely retrieves a session state variable's value."""
    return st.session_state.get(key, default)

  def set(self, key, value):
    """Updates a session state variable's value."""
    st.session_state[key] = value

  def update(self, key, update_func, *args, **kwargs):
    """
    Updates a session state variable using a custom function.
    The update function should accept the current value as its first argument.
    """
    if key in st.session_state:
      current_value = st.session_state[key]
      st.session_state[key] = update_func(current_value, *args, **kwargs)
    else:
      raise KeyError(f"{key} does not exist in session_state.")

  def list_items(self):
    """Lists all items in the session state."""
    return list(st.session_state.items())

  def list_keys(self):
    """Lists all keys in the session state."""
    return list(st.session_state.keys())


state_manager = SessionStateManager()


def reset_evaluations() -> None:
  """
  Reset the evaluations in the session state by initializing it to an empty OrderedDict.
  This is used to clear any existing evaluations stored in the session state.
  """
  state_manager.set("evaluations", OrderedDict())


def reset_eval_pairs() -> None:
  """
  Reset the eval_pairs in the session state by initializing it to an empty OrderedDict.
  This is used to clear any existing eval_pairs stored in the session state.
  """
  state_manager.set("eval_pairs", list())


def _save_human_eval(
    human_eval: models.KeywordEvaluation,
    llm_eval: models.KeywordEvaluation,
) -> None:
  """
  Save the human evaluation alongside the corresponding LLM evaluation.
  Updates the evaluations dictionary and the scored set with the human evaluation,
  and appends the evaluation pair to the eval_pairs list.

  Parameters:
  human_eval (models.KeywordEvaluation): The human evaluation.
  llm_eval (models.KeywordEvaluation): The LLM evaluation.
  """
  evaluation = state_manager.get("evaluations")
  evaluation[human_eval.keyword] = human_eval
  state_manager.set("evaluation", evaluation)

  scored_set = state_manager.get("scored_set")
  scored_set.add(human_eval.keyword)
  state_manager.set("scored_set", scored_set)

  eval_pairs = state_manager.get("eval_pairs")
  eval_pairs.append(
      models.EvaluationPair(
          llm_decision=llm_eval.decision, human_decision=human_eval.decision
      )
  )
  state_manager.set("eval_pairs", eval_pairs)


def define_handler_scoring(
    llm_eval: models.KeywordEvaluation,
    human_agree_with_llm: bool,
):
  """
  Define a handler for scoring, which returns an inner function to handle the logic
  of saving human evaluations based on agreement with LLM evaluations.

  Parameters:
  llm_eval (models.KeywordEvaluation): The LLM evaluation.
  human_agree_with_llm (bool): Indicates if the human agrees with the LLM evaluation.

  Returns:
  Callable: A function that handles the scoring logic.
  """

  def _inner():
    if not human_agree_with_llm:
      state_manager.set(
          "keyword_feedback_eval",
          models.KeywordEvaluation(
              keyword=llm_eval.keyword,
              decision=llm_eval.opposite_decision,
              reason=llm_eval.reason,
          ),
      )
      return

    human_eval = models.KeywordEvaluation(
        keyword=llm_eval.keyword,
        decision=llm_eval.decision,
        reason=llm_eval.reason,
    )
    _save_human_eval(human_eval=human_eval, llm_eval=llm_eval)

  return _inner


def handler_cancel_human_eval() -> Callable:
  """
  Define a handler for canceling a human evaluation.

  Parameters:
  state_manager (SessionStateManager): The session state manager.

  Returns:
  Callable: A function that handles the canceling of human evaluations.
  """

  def _inner():
    state_manager.set("keyword_feedback_eval", None)

  return _inner


def define_handler_save_human_eval(
    llm_eval: models.KeywordEvaluation,
    keyword_feedback_eval: models.KeywordEvaluation,
) -> Callable:
  """
  Define a handler for saving human evaluations, which returns an inner function to handle
  the logic of saving human evaluations.

  Parameters:
  llm_eval (models.KeywordEvaluation): The LLM (Language Model) evaluation.

  Returns:
  Callable: A function that handles the saving of human evaluations.
  """

  def _inner():
    human_eval = models.KeywordEvaluation(
        keyword=llm_eval.keyword,
        decision=keyword_feedback_eval.decision,
        reason=keyword_feedback_eval.reason,
    )
    _save_human_eval(human_eval=human_eval, llm_eval=llm_eval)

  return _inner


def handler_human_decision(event, value) -> None:
  """
  Handler for setting the human decision in the evaluation.

  Parameters:
  event: The event triggered by the UI component.
  value: The value from the UI component.
  """
  kf_eval = state_manager.get("keyword_feedback_eval")
  kf_eval.decision = value.props.value
  state_manager.set("keyword_feedback_eval", kf_eval)


def handler_human_reason(event) -> None:
  """
  Handler for setting the human reason in the evaluation.

  Parameters:
  event: The event triggered by the UI component.
  """
  kf_eval = state_manager.get("keyword_feedback_eval")
  kf_eval.reason = event.target.value
  state_manager.set("keyword_feedback_eval", kf_eval)
