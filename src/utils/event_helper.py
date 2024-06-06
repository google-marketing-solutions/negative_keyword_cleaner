from collections import OrderedDict
from typing import Callable, Any

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
        """Initialize a session state variable if it does not exist."""
        if key not in st.session_state:
            st.session_state[key] = default_value

    def get(self, key, default=None):
        """Safely retrieve a session state variable's value."""
        return st.session_state.get(key, default)

    def set(self, key, value):
        """Update a session state variable's value."""
        st.session_state[key] = value

    def update(self, key, update_func, *args, **kwargs):
        """
        Update a session state variable using a custom function.
        The update function should accept the current value as its first argument.
        """
        if key in st.session_state:
            current_value = st.session_state[key]
            st.session_state[key] = update_func(current_value, *args, **kwargs)
        else:
            raise KeyError(f"{key} does not exist in session_state.")

    def list_items(self):
        """List all items in the session state."""
        return list(st.session_state.items())

    def list_keys(self):
        """List all keys in the session state."""
        return list(st.session_state.keys())


state_manager = SessionStateManager()


def handle_continue_with_context():
    """
    Set the session state for context continuation.
    """
    state_manager.set("context_ready", True)
    state_manager.set("context_open", False)
    state_manager.set("epoch_eval_pairs", [])


def reset_batch_props() -> None:
    """
    Reset the properties related to batch scoring in the session state.
    """
    state_manager.set("batch_scored_keywords", set())
    state_manager.set("keyword_feedback_eval", None)


def handle_selected_customers() -> None:
    """
    Handle the event when customers are selected by resetting batch properties
    and clearing scored keywords.
    """
    reset_batch_props()
    state_manager.set("scored_keywords", None)


def handle_selected_campaigns() -> None:
    """
    Handle the event when campaigns are selected by resetting batch properties
    and clearing scored keywords.
    """
    reset_batch_props()
    state_manager.set("scored_keywords", None)


def handle_continue_with_customers() -> None:
    """
    Handle the continuation with the selected customers by setting the customers
    as ready and closing the customer loader.
    """
    state_manager.set("customers_ready", True)
    state_manager.set("load_customers_open", False)


def handle_continue_with_filters() -> None:
    """
    Set the session state to indicate that filters are ready.
    """
    state_manager.set("filters_ready", True)


def score_batch_evals() -> None:
    """
    Score the batch evaluations. Add the current batch evaluation pairs to the
    epoch evaluation pairs and reset the batch evaluation pairs.
    """
    # Stores batch eval pairs.
    current_batch_eval_pairs = state_manager.get("batch_eval_pairs", None)
    if current_batch_eval_pairs:
        epoch_eval_pairs = state_manager.get("epoch_eval_pairs")
        epoch_eval_pairs.append(current_batch_eval_pairs)
        state_manager.set("batch_eval_pairs", list[models.EvaluationPair]())


def handle_sample_batch() -> None:
    """
    Handle the sampling of a new batch by resetting relevant variables and
    scoring batch evaluations.
    """
    # Resets variables.
    state_manager.set("sample_new_batch", True)
    state_manager.set("load_keywords_open", True)
    state_manager.set("scored_keywords", None)
    state_manager.set("random_state", models.get_random_state(force_new=True))
    reset_batch_props()
    score_batch_evals()


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


def handler_human_decision(event: Any, value: Any) -> None:
    """
    Handler for setting the human decision in the evaluation.

    Parameters:
    event: The event triggered by the UI component.
    value: The value from the UI component.
    """
    kf_eval = state_manager.get("keyword_feedback_eval")
    kf_eval.decision = value.props.value
    state_manager.set("keyword_feedback_eval", kf_eval)


def handler_human_reason(event: Any) -> None:
    """
    Handler for setting the human reason in the evaluation.

    Parameters:
    event: The event triggered by the UI component.
    """
    kf_eval = state_manager.get("keyword_feedback_eval")
    kf_eval.reason = event.target.value
    state_manager.set("keyword_feedback_eval", kf_eval)
