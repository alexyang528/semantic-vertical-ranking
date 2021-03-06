import json
import requests
import streamlit as st
from yext import YextClient
from math import floor
from semantic_vertical_ranking import (
    get_liveapi_response,
    get_new_rank,
    svr,
    svr_result_name,
    dcg_result_name,
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
PRIORITY_FIELDS = ["name", "filter_value", "vertical_id"]


def _keys_exists(element, *keys):
    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


st.sidebar.write("## Vertical Ranking Components")
has_entity_type_filter = st.sidebar.checkbox("Has Entity Type Filter")
has_near_me_filter = st.sidebar.checkbox("Has Near Me Filter")
has_location_filter = st.sidebar.checkbox("Has Location Filter")
score_method = st.sidebar.radio(label="Scoring Method", options=("SVR", "SVR Names", "DCG"))

st.sidebar.write("## General")
bucket = st.sidebar.checkbox("Bucket Similarities")

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

st.sidebar.write("## Semantic Fields")
semantic_fields = {}
for vertical in VERTICALS:
    semantic_fields_str = st.sidebar.text_input("{} (Comma Separated)".format(vertical), key=2)
    semantic_fields[vertical] = [i.strip() for i in semantic_fields_str.split(",") if i]

if YEXT_API_KEY and EXPERIENCE_KEY and QUERY:
    try:
        yext_client = YextClient(YEXT_API_KEY)
        response = get_liveapi_response(QUERY, yext_client, EXPERIENCE_KEY)
        # context = '{"store":"5673","trackingUrl":"../../../ny/brooklyn/144-n-7th-st/track-package"}'
        # url = f'https://liveapi.yext.com/v2/accounts/me/answers/query?input={QUERY}&experienceKey={EXPERIENCE_KEY}&api_key={YEXT_API_KEY}&v=20190101&version=PRODUCTION&locale=en&sessionTrackingEnabled=true&context={context}&referrerPageUrl=&source=STANDARD&jsLibVersion=v1.9.2"'
        # response = requests.get(url).text
        # response = json.loads(response)["response"]
    except:
        raise ValueError("Invalid Experience Key or API Key.")

    # Get general inputs
    modules = [module for module in response["modules"] if module["source"] == "KNOWLEDGE_MANAGER"]
    vertical_ids = [module["verticalConfigId"] for module in modules]
    query_filters = [module["appliedQueryFilters"] for module in modules]

    # Get filter criteria:
    filter_criteria = []
    if has_entity_type_filter:
        entity_type_filter = [any([_keys_exists(filter, "filter", "builtin.entityType") for filter in l if filter]) for l in query_filters]
        filter_criteria.append(entity_type_filter)
    if has_near_me_filter:
        near_me_filter = [any([_keys_exists(filter, "filter", "builtin.location", "$near") for filter in l if filter]) for l in query_filters]
        filter_criteria.append(near_me_filter)
    if has_location_filter:
        location_filter = [any([_keys_exists(filter, "filter", "builtin.location", "$eq") for filter in l if filter]) for l in query_filters]
        filter_criteria.append(location_filter)

    # Get Semantic Vertical Relevance (SVR)
    if score_method == "SVR":
        # Collect inputs
        first_results = [module["results"][0] for module in modules]
        filter_values = [[f_i["displayValue"] for f_i in f] for f in query_filters]

        # Get Semantic Vertical Relevance score and priority field tiebreaker
        scores, max_values, max_fields, embeds = svr(
            QUERY, vertical_ids, first_results, filter_values, vertical_intents, semantic_fields
        )
        is_priority_fields = [
            0 if set(v_max_fields).isdisjoint(PRIORITY_FIELDS) else 1 for v_max_fields in max_fields
        ]

        # Set display template
        template = """
            **Vertical Key:** {}\n
            **Top Result:** {}\n
            **Original Rank:** {}\n
            **Similarity:** {}\n
            **Max Fields:** {}\n
            **Max Field Value:** {}
        """

    # Get Semantic Vertical Relevance (SVR) with just Result Names
    elif score_method == "SVR Names":
        # Collect inputs
        result_names = [
            [
                result.get("data", {}).get("name", None)
                for result in module["results"]
                if result.get("data", {}).get("name", None) is not None
            ]
            for module in modules
        ]

        # Limit to 10 results max per vertical
        result_names = [names[:10] for names in result_names]

        # Get scores
        scores, max_values, max_position, embeds = svr_result_name(QUERY, result_names)
        
        # Set display template
        template = """
            **Vertical Key:** {}\n
            **Original Rank:** {}\n
            **Semantic Vertical Relevance:** {}\n
            **Top Result Name:** {}\n
            **Top Result Position:** {}
        """

    # Get Discount Cumulative Gain (DCG) with Result Names
    elif score_method == "DCG":
        # Collect inputs
        result_names = [
            [
                result.get("data", {}).get("name", None)
                for result in module["results"]
                if result.get("data", {}).get("name", None) is not None
            ]
            for module in modules
        ]

        # Limit to 10 results max per vertical
        result_names = [names[:10] for names in result_names]

        # Get scores
        scores, max_values, max_position, embeds = dcg_result_name(QUERY, result_names)

        # Set display template
        template = """
            **Vertical Key:** {}\n
            **Original Rank:** {}\n
            **Vertical DCG Score:** {}\n
            **Top Result Name:** {}\n
            **Top Result Position:** {}
        """

    # Apply boosts and bucketing
    scores = [
        score + vertical_boosts.get(vertical_id, 0)
        for score, vertical_id in zip(scores, vertical_ids)
    ]
    if bucket:
        scores = [floor(score * 10) / 10 for score in scores]

    # Provide which values should be used for new ranking
    if score_method == "SVR":
        score_criteria = [scores, is_priority_fields]
    else:
        score_criteria = [scores]

    new_ranks = get_new_rank(*filter_criteria, *score_criteria)

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

        if score_method == "SVR":
            left_output = template.format(
                original_module["verticalConfigId"],
                original_module["results"][0]["data"]["name"],
                old_rank,
                scores[old_rank],
                max_fields[old_rank],
                max_values[old_rank],
            )
            right_output = template.format(
                reordered_module["verticalConfigId"],
                reordered_module["results"][0]["data"]["name"],
                f"{old_rank} (?? {delta})",
                scores[new_rank],
                max_fields[new_rank],
                max_values[new_rank],
            )
        else:
            left_output = template.format(
                original_module["verticalConfigId"],
                old_rank,
                scores[old_rank],
                max_values[old_rank],
                max_position[old_rank],
            )
            right_output = template.format(
                reordered_module["verticalConfigId"],
                f"{old_rank} (?? {delta})",
                scores[new_rank],
                max_values[new_rank],
                max_position[new_rank],
            )

        with left_col:
            st.info(left_output)
        with right_col:
            renderer(right_output)
        
        old_rank += 1
    st.write("### Embeddings Calculated: {}".format(embeds))
