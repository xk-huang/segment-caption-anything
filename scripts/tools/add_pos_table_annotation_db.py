import click
import sqlite3
import logging
import spacy
import tqdm
import contextlib
import time
import torch
import os
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISABLE_TIMEER = os.environ.get("DISABLE_TIMER", False)


@contextlib.contextmanager
def timer(timer_name="timer", pbar=None, pos=0):
    if DISABLE_TIMEER:
        return

    start = time.time()
    yield
    end = time.time()
    if pbar is not None:
        pbar.display(f"Time taken in [{timer_name}]: {end - start:.3e}", pos=pos)
    else:
        logger.info(f"Time taken in [{timer_name}]: {end - start:.3e}")


def dict_factory(cursor, row):
    # NOTE: now we will be returning rows as dictionaries instead of tuples
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class RowDataset(torch.utils.data.Dataset):
    def __init__(self, rows, nlp):
        self.nlp = nlp
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def _collate_fn(self, batch):
        # phrases_batches = [row["phrases"].split("\t") for row in batch]
        # NOTE: it is important to use '.' to split the noun chunks, as spacy uses it to determine the token pos.
        all_phrases = [row["phrases"].replace("\t", ". ") for row in batch]
        all_nouns, all_noun_chunks_ls = get_noun_and_noun_chunks(all_phrases, self.nlp)
        if len(all_nouns) != len(batch) or len(all_noun_chunks_ls) != len(batch):
            raise ValueError("Length of all_nouns and batch should be the same")
        for sample, nouns, noun_chunks in zip(batch, all_nouns, all_noun_chunks_ls):
            sample["nouns"] = nouns
            sample["noun_chunks"] = noun_chunks
        return batch


@click.command()
@click.option("--db", help="Path to the database file")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.option("--batch_size", type=int, default=200_000, help="Batch size")
@click.option("--num_workers", type=int, default=12, help="Number of workers")
def main(db: str, debug, batch_size, num_workers):
    nlp = spacy.load("en_core_web_lg", disable=["ner"])

    conn = sqlite3.connect(db)
    # NOTE: now we will be returning rows as dictionaries instead of tuples
    conn.row_factory = dict_factory
    cursor = conn.cursor()

    table_names, table_schemas = get_tables_with_name_and_schema(cursor)

    logger.info(f"Table Names: {table_names}")

    for table_name in table_names:
        if table_name.endswith("_extension"):
            logger.info(f"Skipping table {table_name}, as it is already an extension")
            continue

        extract_pos_to_table(
            nlp, cursor, conn, table_name, debug=debug, batch_size=batch_size, num_workers=num_workers
        )


def extract_pos_to_table(nlp, cursor, conn, table_name, batch_size=200_00, num_workers=8, debug=False):
    if table_name.endswith("_extension"):
        logger.info(f"Skipping table {table_name}, as it is already an extension")
        return

    pos_table_name = f"{table_name}_pos_extension"
    logger.info(f"Extracting POS from table {table_name} to table {pos_table_name}")
    logger.info(f"Batch size: {batch_size}, num_workers: {num_workers}")

    # Drop the extracted_nouns table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {pos_table_name}")
    conn.commit()
    # Create a new table to store the extracted nouns
    cursor.execute(
        f"""  
    CREATE TABLE IF NOT EXISTS {pos_table_name} (  
        region_id INTEGER PRIMARY KEY,  
        nouns TEXT,  
        noun_chunks TEXT,  
        FOREIGN KEY (region_id) REFERENCES {table_name}(region_id)  
    )  
    """
    )
    conn.commit()

    cursor.execute(f"SELECT * FROM {table_name}" + (" LIMIT 10" if debug else ""))
    rows = cursor.fetchall()

    rows_dataset = RowDataset(rows, nlp)
    rows_dataloader = torch.utils.data.DataLoader(
        rows_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=rows_dataset._collate_fn
    )

    # pbar = tqdm.tqdm(rows, total=len(rows))
    # pbar = tqdm.trange(0, len(rows), batch_size)
    pbar = tqdm.tqdm(rows_dataloader, total=len(rows))
    for batch in pbar:
        for sample in batch:
            region_id = sample["region_id"]

            nouns = sample["nouns"]
            nouns_str = "\t".join(nouns)

            noun_chunks = sample["noun_chunks"]
            noun_chunks_str = "\t".join(noun_chunks)

            cursor.execute(
                f"INSERT INTO {pos_table_name} (region_id, nouns, noun_chunks) VALUES (?, ?, ?)",
                (region_id, nouns_str, noun_chunks_str),
            )
        conn.commit()
        pbar.update(batch_size)
    conn.commit()

    logger.info(f"Finished extracting POS from table {table_name} to table {pos_table_name}")


def get_noun_and_noun_chunks(texts, nlp):
    docs = nlp.pipe(texts)
    noun_chunks_ls = []
    nouns = []
    for doc in docs:
        nouns.append(normalize_nouns(doc))
        noun_chunks_ls.append([chunk.text for chunk in doc.noun_chunks])

    return nouns, noun_chunks_ls


def normalize_nouns(doc):
    # NOTE: it is important to use '.' to split the noun chunks, as spacy uses it to determine the token pos.
    normalized_nouns = [
        token.lemma_.lower().strip(string.punctuation)
        for token in doc
        if token.pos_ == "NOUN" or token.pos_ == "PROPN"
    ]
    return normalized_nouns


def get_tables_with_name_and_schema(cursor):
    # Get the list of tables
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")

    # Print the table names and their schema
    table_names = []
    table_schemas = []
    for result in cursor.fetchall():
        table_name, table_schema = result["name"], result["sql"]
        # table_name, table_schema = result
        logger.info(f"Table Name: {table_name}")
        logger.info(f"Table Schema: {table_schema}\n")
        table_names.append(table_name)
        table_schemas.append(table_schema)

    return table_names, table_schemas


if __name__ == "__main__":
    main()
