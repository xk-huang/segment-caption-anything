import re
import json
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def extract_dicts(line):
    return re.findall(r"{[^}]*}", line)


def main(log_file_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(log_file_path)

    # Read log file and extract lines with "eval_loss"
    with open(log_file_path, "r") as log_file:
        log_lines = log_file.readlines()

    eval_loss_pattern = r"{'eval_loss'"
    extracted_dicts = []

    last_epoch = 0
    epoch_offset = 0

    for line in log_lines:
        dicts = extract_dicts(line)
        for d in dicts:
            if re.search(eval_loss_pattern, d):
                # Replace single quotes with double quotes
                d = d.replace("'", '"')
                try:
                    parsed_dict = json.loads(d)
                    current_epoch = parsed_dict["epoch"]

                    # Check if the epoch number has reset
                    if current_epoch < last_epoch:
                        epoch_offset += last_epoch

                    parsed_dict["epoch"] += epoch_offset
                    last_epoch = current_epoch

                    extracted_dicts.append(parsed_dict)
                except json.JSONDecodeError:
                    print(f"Could not parse line: {d}")

    # Save extracted data to a CSV file
    df = pd.DataFrame(extracted_dicts)
    csv_file_path = os.path.join(output_dir, "output.csv")
    df.to_csv(csv_file_path, index=False)

    # Plot "eval_loss" values
    plt.plot(df["epoch"], df["eval_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss vs Epoch")
    plt.grid()

    plot_file_path = os.path.join(output_dir, "plot.png")
    plt.savefig(plot_file_path)
    plt.show()


if __name__ == "__main__":
    log_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        main(log_file_path, output_dir)
    else:
        main(log_file_path)
