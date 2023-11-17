import gradio as gr
from dataclasses import dataclass
import sqlite3
import random
from utils.git_utils.tsv_io import TSVFile
import json
from PIL import Image
import io
import base64
import pandas as pd
import datetime
import os

db_path = "tmp/annotations.db"
# `columns` should be the same as `create_table.sql`
columns = [
    "caption",
    "conf",
    "area",
    "image_area",
    "clip_score",
    "image_id",
    "region_cnt",
    "region_id",
    "is_acceptable",
    "created_at",
]

conn = sqlite3.connect(db_path)
reviews = conn.execute("SELECT * FROM annotations").fetchall()
reviews = pd.DataFrame(reviews, columns=columns)

reviews.to_csv("tmp/annotations.csv", index=False)


"""
# How to get the table names?
import sqlite3

conn = sqlite3.connect('your_database_name.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

# Fetch all rows from the result set and extract the table names
tables_info = cursor.fetchall()
table_names = [info[0] for info in tables_info]

print(table_names)

conn.close()


# How to get the column names?
import sqlite3

conn = sqlite3.connect('your_database_name.db')
cursor = conn.cursor()

table_name = 'your_table_name'
cursor.execute(f'PRAGMA table_info({table_name});')

# Fetch all rows from the result set and extract the column names
columns_info = cursor.fetchall()
column_names = [info[1] for info in columns_info]

print(column_names)

conn.close()
"""
