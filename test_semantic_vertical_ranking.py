import logging
from rich.logging import RichHandler
from rich.progress import Progress


LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import argparse
import json
import os
import re
import requests
import pandas as pd
from pyhive import presto
from scipy.spatial.distance import cosine
from scipy.stats import rankdata
from yext import YextClient

PRESTO_USER = os.getenv("PRESTO_USER")
PRESTO_HOST = os.getenv("PRESTO_HOST")


def _get_data_from_presto(query, PRESTO_USER=PRESTO_USER, PRESTO_HOST=PRESTO_HOST):
    presto_conn = presto.connect(
        host=PRESTO_HOST,
        username=PRESTO_USER,
        port=8889,
        catalog="hive",
        schema="answers2",
    )

    query_export = pd.read_sql(query, presto_conn)
    return query_export


def get_liveapi_response(search_term, yext_client, experience_key):
    results = yext_client.search_answers_universal(query=search_term, experience_key=experience_key)
    response = results.raw_response["response"]
    return response


def _embed(strings: list) -> list:

    embeddings = requests.post(
        "http://cuda-serv01.dev.us2.yext.com:30034/embed", json={"queries": strings, "lang": "en"}
    ).json()["embeddings"]

    embeddings = [[float(i) for i in l] for l in embeddings]
    return embeddings


def _get_embeddings(query, values):

    # Save length of first result values lists
    lengths = [len(_flatten(l)) for l in values]
    values = _flatten(values)

    # Get embeddings of query and all first result values
    embeds = _embed([query] + values)

    # Get embedding of query
    query_embed = embeds[0]
    embeds = embeds[1:]

    # Get embeddings of first result values
    value_embeds = []
    for length in lengths:
        value_embeds.append(embeds[:length])
        embeds = embeds[length:]

    return query_embed, value_embeds


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
    if display_start != 0 and value[display_start - 1] != " ":
        while value[display_start] != " ":
            display_start += 1

    # Adjusting End - Go backward until space
    while value[display_end] != " ":
        display_end -= 1

    return value[display_start:display_end]


def parse_highlighted_fields(vertical_id, first_result):
    # Initialize output lists to the name of the vertical
    matched_values = [_clean_vertical_id(vertical_id)]
    matched_fields = ["vertical_id"]

    # Return just vertical ID if the result JSON is None
    if not first_result:
        LOGGER.error("Received an empty first result JSON for vertical ID: {}.".format(vertical_id))
        return matched_values, matched_fields

    # Try to append the name of the first result by default
    name_value = first_result.get("data", {}).get("name", None)
    if name_value:
        matched_values.append(name_value)
        matched_fields.append("name")

    # Begin JSON parsing highlighted fields
    highlighted_field = first_result.get("highlightedFields", {})

    # For all highlighted fields, pull out the matched substrings and values
    for k, v in highlighted_field.items():
        if v == []:
            continue

        ### For each highlighted field, there are three possible formats:
        # 1) List of {matchedSubstrings / value} pairs - if there are multiple matches
        if type(v) == list:
            try:
                values = [v_i["value"] for v_i in v]
                substrings = [v_i["matchedSubstrings"] for v_i in v]

                processed_values = [
                    get_snippet(value, sub) for value, sub in zip(values, substrings)
                ]

                matched_values.extend(processed_values)
                processed_fields = [
                    k + " (snipped)" if v != pv else k for v, pv in zip(values, processed_values)
                ]
                matched_fields.extend(processed_fields)
            # 2) List of {field: matchedSubstring / value} dicts - if highlighted field is an object
            except:
                values = [v_i[k_i]["value"] for v_i in v for k_i in v_i]
                substrings = [v_i[k_i]["matchedSubstrings"] for v_i in v for k_i in v_i]

                processed_values = [
                    get_snippet(value, sub) for value, sub in zip(values, substrings)
                ]
                matched_values.extend(processed_values)
                processed_fields = [
                    k + " (snipped)" if v != pv else k for v, pv in zip(values, processed_values)
                ]
                matched_fields.extend(processed_fields)
        # 3) Single dict of {matchedSubstrings / value} - if there is just one match
        else:
            processed_values = get_snippet(v["value"], v["matchedSubstrings"])
            matched_values.append(processed_values)
            matched_fields.append(k + " (snipped)" if v["value"] != processed_values else k)

    assert len(matched_values) == len(matched_fields)
    return matched_values, matched_fields


def get_new_vertical_ranks(query, vertical_ids, first_results, boost_vector=None):
    # Get highlighed field values of the top entity result for each vertical
    all_values_and_fields = [
        parse_highlighted_fields(vertical_id, result)
        for vertical_id, result in zip(vertical_ids, first_results)
    ]
    all_values = [i[0] for i in all_values_and_fields]
    all_fields = [i[1] for i in all_values_and_fields]

    # Embed the query and the values of each first result
    query_embed, value_embeds = _get_embeddings(query, all_values)
    embeddings_calculated = len(_flatten(all_values)) + 1

    # Compute similarities between query and matched values
    similarities = [[_similarity(query_embed, v_i) for v_i in v] for v in value_embeds]

    # Boost if a boost vector is provided
    if boost_vector:
        similarities = [[sim + boost for sim in l] for l, boost in zip(similarities, boost_vector)]

    # Get the index of the max similarity, and the corresponding field and value
    max_similarities = [max(l, default=None) for l in similarities]
    idx_max_similarities = [l.index(i) for l, i in zip(similarities, max_similarities)]
    max_values = [l[i] for l, i in zip(all_values, idx_max_similarities)]
    max_fields = [l[i] for l, i in zip(all_fields, idx_max_similarities)]

    # Get the new vertical rankings by sorting on similarities
    new_rank = rankdata(max_similarities, method='ordinal')
    # Flip new rank, so it is highest to lowest similarity
    new_rank = [len(new_rank) - x for x in new_rank]

    assert len(first_results) == len(max_values) == len(max_fields) == len(new_rank)
    return (
        new_rank,
        _flatten(all_fields),
        max_fields,
        max_values,
        max_similarities,
        embeddings_calculated,
    )


def compile_comparison_json(df):
    def _reorder_modules(modules, new_rank):
        return [module for i, module in sorted(zip(new_rank, modules), key=lambda x: x[0])]

    def _replace_modules(response, reordered_modules):
        return {
            "businessId": response["businessId"],
            "failedVerticals": [],
            "modules": reordered_modules,
            "queryId": response["queryId"],
            "searchIntents": response["searchIntents"],
            "locationBias": response["locationBias"],
        }

    # Get new order of modules
    df["modules"] = df["old_results"].apply(lambda response: response["modules"])
    df["reordered_modules"] = [
        _reorder_modules(module, new_rank)
        for module, new_rank in zip(df["modules"], df["new_vertical_rank"])
    ]

    # Compile new liveAPI response format with reordered modules
    df["new_results"] = [
        _replace_modules(old_result, new_modules)
        for old_result, new_modules in zip(df["old_results"], df["reordered_modules"])
    ]

    # Aggregate by answers key
    agg_df = df.groupby("answers_key").agg(
        {"query": list, "old_results": list, "new_results": list}
    )
    agg_df = agg_df.reset_index()

    comparison_json = {}
    for _, row in agg_df.iterrows():
        answersKey = row["answers_key"]
        queries = row["query"]
        oldResults = row["old_results"]
        newResults = row["new_results"]

        comparison_json[answersKey] = [
            {
                "query": query,
                "oldResult": {"query": query, "result": oldResult},
                "newResult": {"query": query, "result": newResult},
            }
            for query, oldResult, newResult in zip(queries, oldResults, newResults)
        ]

    with open("comparison.json", "w") as f:
        json.dump(comparison_json, f)
    return df


def main(args):

    yext_client = YextClient(args.api_key)

    # Initialize DataFrame with queries and answers key
    df = pd.DataFrame(args.search_terms, columns=["query"])
    df["answers_key"] = [args.experience_key] * len(df.index)

    # Get LiveAPI responses for all queries
    LOGGER.info("Getting LiveAPI responses...")
    with Progress() as progress:
        response_progress = progress.add_task("[green]Querying...", total=len(df.index))
        responses = []
        for query in df["query"]:
            response = get_liveapi_response(query, yext_client, args.experience_key)
            responses.append(response)
            progress.update(response_progress, advance=1)
        df["old_results"] = responses

    # Pull out the first entity result for each vertical module returned for each query
    df["vertical_ids"] = df["old_results"].apply(
        lambda response: [i["verticalConfigId"] for i in response["modules"]]
    )

    df["first_results"] = df["old_results"].apply(
        lambda response: [i["results"][0] for i in response["modules"]]
    )

    # Calculate similarities to highlighted fields of first results and rerank verticals
    LOGGER.info("Calculating new vertical ranks...")
    with Progress() as progress:
        rerank_progress = progress.add_task("[green]Calculating...", total=len(df.index))
        new_ranks = []
        all_fields = []
        max_fields = []
        total_embeddings = []
        for q, v, f in zip(df["query"], df["vertical_ids"], df["first_results"]):
            new_rank, all_fs, max_fs, _, _, embeddings = get_new_vertical_ranks(q, v, f)
            new_ranks.append(new_rank)
            all_fields.append(all_fs)
            max_fields.append(max_fs)
            total_embeddings.append(embeddings)
            progress.update(rerank_progress, advance=1)
        df["new_vertical_rank"] = new_ranks
        df["all_fields"] = all_fields
        df["max_fields"] = max_fields
        df["total_embeddings"] = total_embeddings
    LOGGER.info("Done.")

    # Generate comparison JSON for diff tool
    LOGGER.info("Compiling final comparison JSON...")
    compile_comparison_json(df)
    LOGGER.info("Done.")

    LOGGER.info("{} embeddings on average per query.".format(df["total_embeddings"].mean()))

    # Get the max fields
    max_field_counts = dict()
    for field in _flatten(df["max_fields"].values.tolist()):
        max_field_counts[field] = max_field_counts.get(field, 0) + 1
    all_field_counts = dict()
    for field in _flatten(df["all_fields"].values.tolist()):
        all_field_counts[field] = all_field_counts.get(field, 0) + 1
    field_relevance = dict()
    for k in all_field_counts.keys():
        field_relevance[k] = round(max_field_counts.get(k, 0) / all_field_counts[k], 2)

    LOGGER.info("Field relevance (%% of instances where field was max similarity):")
    LOGGER.info(field_relevance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Rerank verticals using semantic similarity to highlighted fields of top results."
        )
    )
    parser.add_argument(
        "-a", "--api_key", type=str, help="API key of the experience to test.", required=True
    )
    parser.add_argument(
        "-e",
        "--experience_key",
        type=str,
        help="Experience key of the experience to test.",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--search_terms",
        nargs="+",
        help="The search terms for which to compute vertical rankings.",
        required=False,
    )
    args = parser.parse_args()

    if not args.search_terms:
        LOGGER.info("Getting top 100 search terms by search volume...")
        query = """
            select tokenizer_normalized_query, count(distinct query_id)
            from log_federator_requests
            where date(dd) > date_add('day', -1, now())
            and answers_key = '{}'
            group by 1
            order by 2 desc
            limit 100
        """.format(
            args.experience_key
        )
        df = _get_data_from_presto(query, PRESTO_USER, PRESTO_HOST)
        args.search_terms = df["tokenizer_normalized_query"].values.tolist()
        LOGGER.info("Done.")

    main(args)
