"""
Library Assistant Bot - Streamlit UI (PS15)
Run with:  streamlit run app.py
"""

import streamlit as st

from bot import extract_keywords, filter_catalogue, load_catalogue, recommend

st.set_page_config(page_title="Library Assistant Bot", page_icon="📚", layout="wide")


@st.cache_data
def get_catalogue(file):
    return load_catalogue(file)


# ---------------- Sidebar: catalogue ----------------
with st.sidebar:
    st.header("📂 Library catalogue")
    uploaded = st.file_uploader("Upload your own catalogue CSV (optional)", type="csv")
    try:
        catalogue = get_catalogue(uploaded if uploaded else "catalogue.csv")
    except Exception as e:
        st.error(f"Could not load catalogue: {e}")
        st.stop()

    st.success(f"{len(catalogue)} books loaded")
    if "topic" in catalogue.columns:
        st.caption("Topics: " + ", ".join(sorted(catalogue["topic"].unique())))
    with st.expander("View full catalogue"):
        st.dataframe(catalogue, hide_index=True)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main: chat ----------------
st.title("📚 Library Assistant Bot")
st.write("Ask for books by **topic** or **author**, e.g. *“books on machine learning”* "
         "or *“anything by Tanenbaum”*.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("What kind of book are you looking for?")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Step 1: keyword filter with pandas
        matches = filter_catalogue(catalogue, query)

        with st.expander(f"🔎 Keyword filter: {len(matches)} matching book(s) sent to the LLM"):
            st.write("Keywords used:", ", ".join(extract_keywords(query)) or "(none)")
            if not matches.empty:
                cols = [c for c in ["title", "author", "topic", "shelf", "available", "match_score"]
                        if c in matches.columns]
                st.dataframe(matches[cols], hide_index=True)

        # Step 2: if nothing matched, don't call the LLM (avoids made-up books)
        if matches.empty:
            answer = ("Sorry, I couldn't find any books matching that in our catalogue. "
                      "Try another topic or author name. Topics we have: "
                      + ", ".join(sorted(catalogue["topic"].unique())) if "topic" in catalogue.columns
                      else "Sorry, no matching books found.")
        else:
            # Step 3: LLM recommends & explains the top 3
            try:
                with st.spinner("Finding the best books for you..."):
                    answer = recommend(query, matches)
            except Exception as e:
                answer = f" Error calling the LLM: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
