import streamlit as st
from yext import YextClient
from semantic_vertical_ranking import (
    get_new_vertical_ranks,
    get_autocomplete_suggestions,
    get_liveapi_response,
)

"""
# Vertical Ranking 2.0 Prototype
This app is an interactive demo of a new vertical ranking approach, including vertical score using 
semantic similarity of top results, vertical boosts, and vertical intents.
"""

# BUSINESS_ID = st.text_input("Business ID")
YEXT_API_KEY = st.text_input("API Key")
EXPERIENCE_KEY = st.text_input("Experience Key")
QUERY = st.text_input("Query")
VERTICALS = st.text_input("Vertical Keys for Boosting and Intents (Comma Separated)")

VERTICALS = [v.strip() for v in VERTICALS.split(",") if v]

st.sidebar.write("## Vertical Boosts")
vertical_boosts = {}
for vertical in VERTICALS:
    vertical_boosts[vertical] = st.sidebar.slider(
        label=vertical, value=0.0, min_value=-1.0, max_value=1.0, step=0.05
    )

st.sidebar.write("## Vertical Intents")
vertical_intents = {}
for vertical in VERTICALS:
    vertical_intents_str = st.sidebar.text_input("{} (Comma Separated)".format(vertical), key=1)
    vertical_intents[vertical] = [i.strip() for i in vertical_intents_str.split(",") if i]

st.sidebar.write("## Limit to Semantic Fields")
semantic_fields = {}
for vertical in VERTICALS:
    semantic_fields_str = st.sidebar.text_input("{} (Comma Separated)".format(vertical), key=2)
    semantic_fields[vertical] = [i.strip() for i in semantic_fields_str.split(",") if i]

# st.sidebar.write("## Include Autocomplete?")
# include_autocomplete = {}
# for vertical in VERTICALS:
#     autocomplete_bool = st.sidebar.checkbox(label=vertical)
#     include_autocomplete[vertical] = autocomplete_bool


if YEXT_API_KEY and EXPERIENCE_KEY and QUERY:
    try:
        yext_client = YextClient(YEXT_API_KEY)
        response = get_liveapi_response(QUERY, yext_client, EXPERIENCE_KEY)
    except:
        raise ValueError("Invalid Experience Key or API Key.")
    modules = [module for module in response["modules"] if module["source"] == "KNOWLEDGE_MANAGER"]
    first_results = [
        module["results"][0]
        for module in modules
    ]
    vertical_ids = [
        module["verticalConfigId"]
        for module in modules
    ]
    query_filters = [
        module["appliedQueryFilters"]
        for module in modules
    ]
    filter_values = [[f_i["displayValue"] for f_i in f] for f in query_filters]

    # Get autocomplete suggestions for each vertical key
    # for vertical in include_autocomplete:
    #     if include_autocomplete[vertical]:
    #         first_word = QUERY.split(" ")[0]
    #         print(vertical)
    #         prompts = get_autocomplete_suggestions(
    #             BUSINESS_ID, EXPERIENCE_KEY, YEXT_API_KEY, vertical, first_word
    #         )
    #         print(prompts)

    #         if vertical in vertical_intents:
    #             vertical_intents[vertical].extend(prompts)
    #         else:
    #             vertical_intents[vertical] = prompts

    #         print(vertical_intents[vertical])

    # Get new vertical ranks and fields/ values driving change
    new_ranks, max_fields, max_values, max_similarities, embeddings = get_new_vertical_ranks(
        QUERY,
        vertical_ids,
        first_results,
        filter_values,
        semantic_fields,
        vertical_intents,
        vertical_boosts,
    )

    # Remove line breaks from values for better presentation
    max_values = [i.replace("\n", " ") if i else None for i in max_values]

    left_col, right_col = st.columns(2)
    with left_col:
        st.write("## Original Results")
    with right_col:
        st.write("## Reordered Results")

    old_rank = 0
    while old_rank in new_ranks:
        # Get the index of the new item at old_rank
        new_rank = new_ranks.index(old_rank)

        original_module = modules[old_rank]
        reordered_module = modules[new_rank]
        delta = new_rank - old_rank

        if delta > 0:
            renderer = st.success
        elif delta < 0:
            renderer = st.error
        elif delta == 0:
            renderer = st.warning

        with left_col:
            st.info(
                f"""
            **Vertical Key:** {original_module['verticalConfigId']}\n
            **Top Result:** {original_module['results'][0]['data']['name']}\n
            **Original Rank:** {old_rank}\n
            **Similarity:** {max_similarities[old_rank]}\n
            **Max Field:** {max_fields[old_rank]}\n
            **Max Field Value:** {max_values[old_rank]}
            """
            )
        with right_col:
            renderer(
                f"""
            **Vertical Key:** {reordered_module['verticalConfigId']}\n
            **Top Result:** {reordered_module['results'][0]['data']['name']}\n
            **Reordered Rank:** {old_rank} (Δ{delta})\n
            **Similarity:** {sorted(max_similarities, reverse=True)[old_rank]}\n
            **Max Field:** {max_fields[new_rank]}\n
            **Max Field Value:** {max_values[new_rank]}
            """
            )
        old_rank += 1
    st.write("### Embeddings Calculated: {}".format(embeddings))
