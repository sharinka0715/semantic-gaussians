"""Read the scannetv2-labels.combined.tsv and convert the label to other ids"""
import os
import csv


def read_label_mapping(filename, label_from="id", label_to="nyu40id"):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])

    # if ints convert
    def represents_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping
