import sqlite3
import logging
import gradio as gr
import time
import contextlib
import os
import pandas as pd
import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DISABLE_TIMEER = os.environ.get("DISABLE_TIMER", False)
DEBUG = os.environ.get("DEBUG", False)


@contextlib.contextmanager
def timer(timer_name="timer", pbar=None, pos=0):
    if DISABLE_TIMEER:
        return

    start = time.time()
    yield
    end = time.time()
    if pbar is not None:
        pbar.display(f"Time taken in [{timer_name}]: {end - start:.2f}", pos=pos)
    else:
        logger.info(f"Time taken in [{timer_name}]: {end - start:.2f}")


def get_tables_with_name_and_schema(cursor):
    # Get the list of tables
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")

    # Print the table names and their schema
    table_names = []
    table_schemas = []
    for result in cursor.fetchall():
        table_name, table_schema = result["name"], result["sql"]
        table_names.append(table_name)
        table_schemas.append(table_schema)

    return table_names, table_schemas


def load_rows(cursor, table_name):
    print(f"Loading table: {table_name}")
    pos_table_name = table_name + "_pos_extension"
    cursor.execute(
        f"""  
        SELECT {table_name}.region_id, {table_name}.phrases, {pos_table_name}.nouns, {pos_table_name}.noun_chunks
        FROM {table_name}
        JOIN {pos_table_name} ON {table_name}.region_id = {pos_table_name}.region_id
    """
        + ("LIMIT 10" if DEBUG else "")
    )

    rows = cursor.fetchall()
    logger.info(f"Finished loading table: {table_name} with {len(rows)} rows")
    return rows


def dict_factory(cursor, row):
    # NOTE: now we will be returning rows as dictionaries instead of tuples
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


DB = "tmp/annotation_db/objects365-local/annotations.db"


@click.command()
@click.option("--db", help="Path to the database file", default=DB)
def main(db):
    def load_db(db):
        if not os.path.exists(db):
            raise ValueError(f"Database file {db} does not exist.")
        conn = sqlite3.connect(db)

        conn.row_factory = dict_factory

        cursor = conn.cursor()
        table_names, _ = get_tables_with_name_and_schema(cursor)
        table_names = list(filter(lambda x: not x.endswith("_extension"), table_names))

        logger.info(f"Table Names: {table_names}")

        rows_ls = []
        for table_name in table_names:
            with timer():
                rows = load_rows(cursor, table_name)
                rows_ls.append(rows)

        conn.close()
        return table_names, rows_ls

    class DataFrameWithBatchSlider:
        def __init__(self, rows, num_samples, batch_size=10):
            self.rows = rows

            with gr.Row():
                self.num_samples = gr.Textbox(
                    lines=1, value=str(num_samples), label="Number of samples", interactive=False
                )
                self.batch_idx = gr.Slider(
                    minimum=0, maximum=num_samples, step=batch_size, value=0, label="batch_idx", interactive=True
                )
                self.batch_size = gr.Slider(
                    minimum=1, maximum=num_samples, step=1, value=batch_size, label="batch_size", interactive=True
                )

            self.data_frame = gr.DataFrame(pd.DataFrame(rows[0 : 0 + batch_size]))

            self.update_slider(self.batch_idx)
            self.update_slider(self.batch_size)

        def update_data_frame(self, batch_idx, batch_size):
            new_rows = self.rows[batch_idx : batch_idx + batch_size]
            return pd.DataFrame(new_rows)

        def update_slider(self, obj):
            # NOTE: This is how gr.update works. It takes an input value, and applies it to the output object
            handle = obj.change(lambda value: gr.update(value=value), inputs=[obj], outputs=[obj])
            # NOTE: if it is batch_size, we need to upate the step of batch_idx
            if obj is self.batch_size:
                handle.then(lambda step: gr.update(step=step), inputs=[obj], outputs=[self.batch_idx])
            handle.then(
                self.update_data_frame,
                inputs=[self.batch_idx, self.batch_size],
                outputs=[self.data_frame],
            )

    with gr.Blocks() as app:
        db_tb = gr.Textbox(lines=1, value=db, label="Input database path")

        table_names, rows_ls = load_db(db)
        for table_name, rows in zip(table_names, rows_ls):
            with gr.Accordion(label=table_name):
                num_samples = len(rows)
                DataFrameWithBatchSlider(rows, num_samples)

    app.launch()


if __name__ == "__main__":
    main()
