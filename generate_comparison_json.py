import logging
import requests
from rich.logging import RichHandler
from rich.progress import Progress

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import argparse
import json
import os
import pandas as pd
from math import floor
from pyhive import presto
from yext import YextClient
from semantic_vertical_ranking import dcg_result_name, svr, get_new_rank, get_liveapi_response

PRESTO_USER = os.getenv("PRESTO_USER")
PRESTO_HOST = os.getenv("PRESTO_HOST")

PRIORITY_FIELDS = ["name", "filter_value", "vertical_id"]


def _keys_exists(element, *keys):
    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def _flatten(values):
    out = []
    for value in values:
        if isinstance(value, list):
            out.extend(_flatten(value))
        else:
            out.append(value)
    return out


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


def get_comparison_json(query, response, modules, new_ranks):
    def _reorder(old_modules, new_rank):
        return [module for i, module in sorted(zip(new_rank, old_modules), key=lambda x: x[0])]

    def _replace(response, new_modules):
        return {
            "businessId": response["businessId"],
            "failedVerticals": [],
            "modules": new_modules,
            "queryId": response["queryId"],
            "searchIntents": response["searchIntents"],
            "locationBias": response["locationBias"],
            "directAnswer": response.get("directAnswer", ""),
        }

    return {
        "query": query,
        "oldResult": {"query": query, "result": _replace(response, modules)},
        "newResult": {"query": query, "result": _replace(response, _reorder(modules, new_ranks))},
    }


def get_top_search_terms_from_presto(answers_key, business_id, count=500):
    LOGGER.info(f"Getting top {count} search terms by search volume...")
    query = f"""
        select tokenizer_normalized_query, count(distinct query_id)
        from log_federator_requests
        where date(dd) > date_add('day', -7, now())
        and answers_key = '{answers_key}'
        and business_id = {business_id}
        and tokenizer_normalized_query is not null
        and tokenizer_normalized_query != ' '
        group by 1
        order by 2 desc
        limit {count}
    """
    df = _get_data_from_presto(query, PRESTO_USER, PRESTO_HOST)
    search_terms = df["tokenizer_normalized_query"].values.tolist()
    LOGGER.info("Done. Received {} search terms from Presto.".format(len(search_terms)))
    return search_terms


def get_applied_filters(query_filters):
    entity_type_filter = [
        any([_keys_exists(filter, "filter", "builtin.entityType") for filter in l if filter])
        for l in query_filters
    ]
    near_me_filter = [
        any([_keys_exists(filter, "filter", "builtin.location", "$near") for filter in l if filter])
        for l in query_filters
    ]
    location_filter = [
        any([_keys_exists(filter, "filter", "builtin.location", "$eq") for filter in l if filter])
        for l in query_filters
    ]

    return [entity_type_filter, near_me_filter, location_filter]


def main(args):

    # Fetch top [limit] search terms by search volume from Presto
    search_terms = get_top_search_terms_from_presto(
        args.experience_key, args.business_id, args.limit
    )

    # Initialize Yext client
    yext_client = YextClient(args.api_key)

    # Get vertical boosts and intents if file provided
    vertical_boosts = {}
    vertical_intents = {}
    if args.boost_file:
        vertical_boosts = json.load(open(args.boost_file))
    if args.intents_file:
        vertical_intents = json.load(open(args.intents_file))

    # Initialize final comparison JSON
    comparison_json = {}
    comparison_json[args.experience_key] = []

    # Get LiveAPI responses for all queries
    LOGGER.info("Reranking...")
    with Progress() as progress:
        rerank_progress = progress.add_task("[green]Reranking...", total=len(search_terms))
        for query in search_terms:
            # Get Live API response for each search term
            response = get_liveapi_response(query, yext_client, args.experience_key)

            # for UPS - context to specify a single store
            # context = '{"store":"5673","trackingUrl":"../../../ny/brooklyn/144-n-7th-st/track-package"}'
            # url = f'https://liveapi.yext.com/v2/accounts/me/answers/query?input={query}&experienceKey={args.experience_key}&api_key={args.api_key}&v=20190101&version=PRODUCTION&locale=en&sessionTrackingEnabled=true&context={context}&referrerPageUrl=&source=STANDARD&jsLibVersion=v1.9.2"'
            # response = requests.get(url).text
            # response = json.loads(response)["response"]

            # Keep just KM modules
            modules = [
                module for module in response["modules"] if module["source"] == "KNOWLEDGE_MANAGER"
            ]

            # Pull out top results, vertical IDs, and filter values for each KM module returned
            first_results = [module["results"][0] for module in modules]
            vertical_ids = [module["verticalConfigId"] for module in modules]
            query_filters = [module["appliedQueryFilters"] for module in modules]
            filter_values = [[f_i["displayValue"] for f_i in f] for f in query_filters]
            result_names = [
                [
                    result.get("data", {}).get("name", None)
                    for result in module["results"]
                    if result.get("data", {}).get("name", None) is not None
                ]
                for module in modules
            ]
            result_names = [names[:10] for names in result_names]

            # Get applied filter booleans
            filter_criteria = []
            if args.filters:
                filter_criteria = get_applied_filters(query_filters)

            # Get DCG score if selected
            if args.dcg:
                scores = dcg_result_name(query, result_names)[0]
            # Or get SVR scores, and whether scores are on priority fields
            else:
                scores, _, max_fields, _ = svr(
                    query, vertical_ids, first_results, filter_values, vertical_intents, {}
                )
                is_priority_fields = [
                    0 if set(fields).isdisjoint(PRIORITY_FIELDS) else 1 for fields in max_fields
                ]

            # Apply boosts and bucketing to scores
            scores = [
                score + vertical_boosts.get(vertical_id, 0)
                for score, vertical_id in zip(scores, vertical_ids)
            ]
            if args.bucket:
                scores = [floor(score * 10) / 10 for score in scores]

            # Provide which values should be used for new ranking
            if args.dcg:
                score_criteria = [scores]
            else:
                score_criteria = [scores, is_priority_fields]

            # Get new rank of verticals for query
            new_ranks = get_new_rank(*filter_criteria, *score_criteria)

            query_json = get_comparison_json(query, response, modules, new_ranks)
            comparison_json[args.experience_key].append(query_json)

            progress.update(rerank_progress, advance=1)

    with open(args.output, "w") as f:
        json.dump(comparison_json, f)

    LOGGER.info("Done.")


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
        "-b",
        "--business_id",
        type=int,
        help="Business ID of the experience to test.",
        required=True,
    )
    parser.add_argument(
        "--bucket",
        action="store_true",
        help="Whether or not to bucket similarities to 1/10th place.",
        default=False,
    )
    parser.add_argument(
        "--filters",
        action="store_true",
        help="Whether or not to include entity type, location, and near me filters.",
        default=False,
    )
    parser.add_argument(
        "--dcg",
        action="store_true",
        help="Whether or not to use Discount Cumulative Gain method.",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Number of search terms to include.",
        required=False,
        default=500,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The name of the output JSON file.",
        default="comparison.json",
    )
    parser.add_argument(
        "--boost_file",
        type=str,
        help="The name of the JSON file to read boosts dict from.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--intents_file",
        type=str,
        help="The name of the JSON file to read intents dict from.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--semantic_file",
        type=str,
        help="The name of the JSON file to read semantic fields dict from.",
        required=False,
        default=None,
    )
    args = parser.parse_args()
    LOGGER.info(args)

    main(args)
