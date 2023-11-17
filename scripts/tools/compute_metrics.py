import click
import json
import evaluate
import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("json_file", type=click.Path(exists=True))
@click.option("--split")
def compute_metrics(json_file, split):
    compute_metrics_func = evaluate.load("meteor")
    with open(json_file) as f:
        data = json.load(f)
    refs = []
    preds = []
    num_heads = None

    logger.info(f"Loading {json_file}")
    for sample in filter(lambda x: x["split"] == split, tqdm.tqdm(data)):
        refs_ = sample["references"]
        preds_ = sample["candidates"]

        if num_heads is None:
            num_heads = max(len(refs_), len(preds_))

        refs.extend(refs_ + [refs_[-1]] * (num_heads - len(refs_)))
        preds.extend(preds_ + [preds_[-1]] * (num_heads - len(preds_)))

    logger.info(f"Computing metrics for {json_file}")
    metrics = compute_metrics_func.compute(predictions=preds, references=refs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    compute_metrics()
