import streamlit as st


def display_faq_component():
  st.header("FAQ")
  with st.sidebar.expander("How does AI Student work?"):
    st.markdown(
        """We help you build the most efficient prompt to solve your marketing
            maintenance task. Unlocking the full capabilities of LLMs."""
    )
  with st.sidebar.expander("Is my data safe?"):
    st.markdown("""Yes, your data is safe.
        All information that is sent to the LLM is done privately, so that it can’t be
        used to train future versions of the models.""")
  with st.sidebar.expander("Does the tool hallucinate?"):
    st.markdown(
        """When a generative AI has to answer a question, it will write down the
            most statistically probable answer, despite the truth. This is what can lead to
            hallucinations.
            By giving the AI a source of information (here the positive and negative prompt),
            it will help improve the relevance of the response."""
    )
  with st.sidebar.expander("How much does it cost?"):
    st.markdown(
        "We estimate that the maximum cost to score 100,000 negative keywords "
        "is $20."
    )
