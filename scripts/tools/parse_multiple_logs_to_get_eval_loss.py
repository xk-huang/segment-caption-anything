import re
import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob


def extract_dicts(line):
    return re.findall(r"{[^}]*}", line)


def process_log_file(log_file_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(log_file_path)

    extracted_dicts = []

    with open(log_file_path, "r") as log_file:
        log_lines = log_file.readlines()

    eval_loss_pattern = r"{'eval_loss'"
    last_epoch = 0
    epoch_offset = 0

    for line in log_lines:
        dicts = extract_dicts(line)
        for d in dicts:
            if re.search(eval_loss_pattern, d):
                d = d.replace("'", '"')
                try:
                    parsed_dict = json.loads(d)
                    current_epoch = parsed_dict["epoch"]

                    if current_epoch < last_epoch:
                        epoch_offset += last_epoch

                    parsed_dict["epoch"] += epoch_offset
                    last_epoch = current_epoch

                    extracted_dicts.append(parsed_dict)
                except json.JSONDecodeError:
                    print(f"Could not parse line: {d}")

    df = pd.DataFrame(extracted_dicts)
    csv_file_path = os.path.join(output_dir, "output.csv")
    df.to_csv(csv_file_path, index=False)

    plt.plot(df["epoch"], df["eval_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss vs Epoch")
    plt.grid()

    plot_file_path = os.path.join(output_dir, "plot.png")
    plt.savefig(plot_file_path)
    plt.clf()


def process_directory(source_dir, target_log_name):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file == target_log_name:
                log_file_path = os.path.join(root, file)
                process_log_file(log_file_path)


import pandas as pd
import matplotlib.pyplot as plt
import os


def gather_results(source_dir, target_csv_name):
    all_results = []

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file == target_csv_name:
                csv_file_path = os.path.join(root, file)
                df = pd.read_csv(csv_file_path)
                all_results.append((os.path.dirname(csv_file_path), df))

    return all_results


def plot_comparison(all_results, output_dir):
    for csv_dir, df in all_results:
        plt.plot(df["epoch"], df["eval_loss"], label=csv_dir.split("/")[-1])

    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss vs Epoch")
    plt.grid()
    plt.legend()

    plot_file_path = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(plot_file_path)
    plt.show()


if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_log_name = sys.argv[2]

    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    else:
        output_dir = source_dir

    process_directory(source_dir, target_log_name)
    all_results = gather_results(source_dir, "output.csv")
    plot_comparison(all_results, output_dir)
