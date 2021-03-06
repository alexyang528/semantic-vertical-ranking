import logging

LOGGER = logging.getLogger(__name__)

import re
import requests
import pandas as pd
from math import log
from scipy.spatial.distance import cosine


def get_liveapi_response(search_term, yext_client, experience_key):
    results = yext_client.search_answers_universal(query=search_term, experience_key=experience_key)
    response = results.raw_response["response"]
    return response


def _embed(strings: list) -> list:

    embeddings = requests.post(
        "http://cuda-serv01.dev.us2.yext.com:30035/embed", json={"queries": strings, "lang": "en"}
    ).json()["embeddings"]

    embeddings = [[float(i) for i in l] for l in embeddings]
    return embeddings


def _get_embeddings(query, values):

    # Save length of first result values lists
    lengths = [len(_flatten(l)) for l in values]
    values = _flatten(values)

    # Strip and normalize all values (remove punctuation, etc.)
    query = re.sub("[^\w\s]", " ", query)
    values = [re.sub("[^\w\s]", " ", value) for value in values]

    # Get only unique values to embed
    unique = list(set([query] + values))

    # Get embeddings of query and all first result values
    embeds = _embed(unique)

    # Map embeddings back to original unique values
    embeds_dict = dict(zip(unique, embeds))

    # Get embedding of query
    query_embed = embeds_dict[query]

    # Get embeddings of first result values
    flat_value_embeds = [embeds_dict[value] for value in values]

    # Unflatten values back into list of list
    value_embeds = []
    for length in lengths:
        value_embeds.append(flat_value_embeds[:length])
        flat_value_embeds = flat_value_embeds[length:]

    return query_embed, value_embeds, len(unique)


def _similarity(embed_a, embed_b):
    return 1 - cosine(embed_a, embed_b)


def _flatten(values):
    out = []
    for value in values:
        if isinstance(value, list):
            out.extend(_flatten(value))
        else:
            out.append(value)
    return out


def _clean_vertical_id(vertical_id):
    # Convert camel case into space-separated words
    cleaned_vertical_id = re.sub("([a-z])([A-Z])", "\g<1> \g<2>", vertical_id)
    # Replace - and _
    cleaned_vertical_id = cleaned_vertical_id.replace("-", " ")
    cleaned_vertical_id = cleaned_vertical_id.replace("_", " ")
    # Lowercase
    cleaned_vertical_id = cleaned_vertical_id.lower()
    return cleaned_vertical_id


def get_snippet(value, matched_subs, chars_before=50, chars_after=50, use_dense=True):
    if not matched_subs:
        return value[:chars_after]
    if len(value) < (chars_before + chars_after):
        return value

    best_substring = matched_subs[0]

    # Select snippet by density strategy
    if use_dense:
        most_subs = 1
        for i, matched_sub in enumerate(matched_subs):
            start_offset = matched_sub["offset"]
            remaining_subs = matched_subs[i:]
            inrange_subs = 0
            for sub in remaining_subs:
                if start_offset + sub["offset"] + sub["length"] > chars_after:
                    inrange_subs += 1
                else:
                    break
            if inrange_subs > most_subs:
                best_substring = matched_sub
                most_subs = inrange_subs

    offset = best_substring["offset"]
    display_start = max(offset - chars_before, 0)
    display_end = min(offset + chars_after, len(value) - 1)

    # Adjusting Start - Go forward until space
    if (
        display_start != 0
        and value[display_start - 1] != " "
        and " " in value[display_start:display_end]
    ):
        while value[display_start] != " ":
            display_start += 1

    if display_end != len(value) - 1 and " " in value[display_start:display_end]:
        # Adjusting End - Go backward until space
        while value[display_end] != " ":
            display_end -= 1

    return value[display_start:display_end]


def _get_semantic_fields(data, semantic_fields):

    # Get the value of semantic fields from data returned in LiveAPI
    fields_and_values = [
        (data[semantic_field], semantic_field)
        for semantic_field in semantic_fields
        if semantic_field in data
    ]
    values = [i[0] for i in fields_and_values]
    fields = [i[1] for i in fields_and_values]

    return values, fields


def _parse_value_recursively(field, value):

    # 1) Single dict of {matchedSubstrings / value} - if there is just one match
    if type(value) == dict and set(value.keys()) == set(["value", "matchedSubstrings"]):
        processed_value = get_snippet(value["value"], value["matchedSubstrings"])
        processed_field = field + " (snipped)" if value["value"] != processed_value else field
        return [processed_value], [processed_field]

    # 2) Dict of {field: {matchedSubstrings / value}} - if field is an object
    if type(value) == dict:
        values_and_fields = [_parse_value_recursively(f, v) for f, v in value.items()]
        processed_values = _flatten([i[0] for i in values_and_fields])
        processed_fields = _flatten([i[1] for i in values_and_fields])
        return processed_values, processed_fields

    # 3) List of {matchedSubstrings / value} or {field: {matchedSubstring / value}}
    if type(value) == list:
        values_and_fields = [_parse_value_recursively(field, v) for v in value]
        processed_values = _flatten([i[0] for i in values_and_fields])
        processed_fields = _flatten([i[1] for i in values_and_fields])
        return processed_values, processed_fields


def parse_highlighted_fields(
    vertical_id, first_result, filter_values, vertical_intents, semantic_fields
):

    matches = {}

    # Initialize with vertical intents
    for intent in vertical_intents:
        matches[intent] = matches.get(intent, []) + ["vertical_intent"]

    # Add NLP filter values to values to consider
    for filter in filter_values:
        matches[filter] = matches.get(filter, []) + ["filter_value"]

    # Try to append the name of the first result by default
    name_value = first_result.get("data", {}).get("name", None)
    if name_value:
        matches[name_value] = matches.get(name_value, []) + ["name"]

    # Append the name of the vertical ID by default
    clean_vertical_id = _clean_vertical_id(vertical_id)
    matches[clean_vertical_id] = matches.get(clean_vertical_id, []) + ["vertical_id"]

    # Return just vertical intents if the result JSON is None
    if not first_result:
        LOGGER.warning("Empty first result JSON for vertical ID: {}.".format(vertical_id))
        return matches

    # If semantic fields are provided, only pull values of semantic fields
    if semantic_fields:
        data = first_result.get("data", {})
        semantic_values, semantic_fields_found = _get_semantic_fields(data, semantic_fields)
        for v, f in zip(semantic_values, semantic_fields_found):
            matches[v] = matches.get(v, []) + [f]

    # Otherwise, recursively get field, value pairs from highlightedFields
    else:
        highlights = first_result.get("highlightedFields", {})
        highlighted_values, highlighted_fields = _parse_value_recursively(None, highlights)
        for v, f in zip(highlighted_values, highlighted_fields):
            matches[v] = matches.get(v, []) + [f]

    return matches


def svr(query, vertical_ids, first_results, filter_values, vertical_intents={}, semantic_fields={}):
    # Get highlighed field values of the top entity result for each vertical
    all_matches = [
        parse_highlighted_fields(
            vertical_id,
            result,
            filters,
            vertical_intents.get(vertical_id, []),
            semantic_fields.get(vertical_id, [])
        )
        for vertical_id, result, filters in zip(vertical_ids, first_results, filter_values)
    ]
    all_values = [list(vertical_matches.keys()) for vertical_matches in all_matches]

    # Embed the query and the values of each first result
    query_embed, value_embeds, embeddings_calculated = _get_embeddings(query, all_values)

    # Compute similarities between query and matched values
    similarities = [[_similarity(query_embed, v_i) for v_i in v] for v in value_embeds]

    # Get the max similarity
    max_similarities = [max(l, default=-1) for l in similarities]

    # Get the index of the max similarity
    idx_max_similarities = [
        l.index(i) if i != -1 else -1 for l, i in zip(similarities, max_similarities)
    ]
    # Get value and field(s) providing the top similarity value
    max_values = [l[i] if i != -1 else None for l, i in zip(all_values, idx_max_similarities)]
    max_fields = [
        vertical_matches.get(value, None)
        for vertical_matches, value in zip(all_matches, max_values)
    ]

    return max_similarities, max_values, max_fields, embeddings_calculated


def svr_result_name(query, result_names):

    # Embed the query and the values of each first result
    query_embed, value_embeds, embeddings_calculated = _get_embeddings(query, result_names)

    # Compute similarities between query and matched values
    similarities = [[_similarity(query_embed, v_i) for v_i in v] for v in value_embeds]

    # Get the max similarity and index of max similarity
    max_similarities = [max(l, default=-1) for l in similarities]
    max_position = [
        l.index(i) if i != -1 else -1 for l, i in zip(similarities, max_similarities)
    ]
    # Get result name and position providing the top similarity value
    max_values = [l[i] if i != -1 else None for l, i in zip(result_names, max_position)]

    return max_similarities, max_values, max_position, embeddings_calculated


def dcg_result_name(query, result_names):

    def dcg_x(scores, x):
        return sum([score / log(i+2, 2) for i, score in enumerate(scores[:x])])

    # Embed the query and the values of each first result
    query_embed, value_embeds, embeddings_calculated = _get_embeddings(query, result_names)

    # Compute similarities between query and matched values
    similarities = [[_similarity(query_embed, v_i) for v_i in v] for v in value_embeds]

    # Get the DCG (discount cumulative gain) for first 10 results per vertical
    dcg = [dcg_x(similarity, 10) for similarity in similarities]

    # Get the max similarity and index of max similarity
    max_similarities = [max(l, default=-1) for l in similarities]
    max_position = [
        l.index(i) if i != -1 else -1 for l, i in zip(similarities, max_similarities)
    ]
    # Get result name and position providing the top similarity value
    max_values = [l[i] if i != -1 else None for l, i in zip(result_names, max_position)]

    return dcg, max_values, max_position, embeddings_calculated


def get_new_rank(*ranking_criteria):

    # Return empty ranking list if length of ranking criteria is 0
    if any([len(arg) for arg in ranking_criteria]) == 0:
        return []

    # Check that all ranking criteria are the same length
    if not len(set([len(arg) for arg in ranking_criteria])) == 1:
        LOGGER.error("Args for new ranking must all be equal length.")
        raise ValueError

    df = pd.DataFrame(ranking_criteria).T

    df["rank"] = df.sort_values(df.columns.to_list(), ascending=False)[0].rank(
        method="first", ascending=False
    )

    return [int(i) - 1 for i in df["rank"].to_list()]
