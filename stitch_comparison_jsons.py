import logging
from rich.logging import RichHandler

LOG_FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=LOG_FORMAT, datefmt="[%X] ", handlers=[RichHandler()])
LOGGER = logging.getLogger(__name__)

import argparse
import json


def stitch_comparison_jsons(args):
    LOGGER.info("Stitching before and after JSONs...")
    
    with open(args.before_json, "r") as f:
        before_json = json.load(f)
        before_queries = [i["query"] for i in before_json[args.experience_key]]

    with open(args.after_json, "r") as f:
        after_json = json.load(f)
        after_queries = [i["query"] for i in after_json[args.experience_key]]

    overlapping_queries = list(set(before_queries) & set(after_queries))
    LOGGER.info(f"{len(overlapping_queries)} overlapping queries found.")

    final_json = {}
    results = []

    LOGGER.info("Generating stitched JSON...")
    for query in overlapping_queries:
        before_index = before_queries.index(query)
        after_index = after_queries.index(query)

        result = {
            "query": query,
            "oldResult": before_json[args.experience_key][before_index]["oldResult"],
            "newResult": after_json[args.experience_key][after_index]["newResult"],
        }

        results.append(result)

    final_json[args.experience_key] = results
    with open(args.output, "w") as f:
        json.dump(final_json, f)
    LOGGER.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Stitch together two comparison JSONs.")
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
        "--before_json",
        type=str,
        help="The name of the JSON file to use for old results.",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--after_json",
        type=str,
        help="The name of the JSON file to use for new results.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The name of the output JSON file.",
        default="stitched.json"
    )
    args = parser.parse_args()
    LOGGER.info(args)

    stitch_comparison_jsons(args)
