import csv


def read_data(filename):
    data = []

    with open(filename, 'r', encoding='utf-16') as infile:
        tsv_reader = csv.reader(infile, delimiter="\n")
        for row in tsv_reader:
            for r in row:
                dialect, datastring = r.split('\t')
                data.append(datastring)
    return data


def read_data_with_labels(filename):
    data = []
    labels = []
    with open(filename, 'r', encoding='utf-16') as infile:
        tsv_reader = csv.reader(infile, delimiter="\n")
        for row in tsv_reader:
            for r in row:
                dialect, datastring = r.split('\t')
                labels.append(dialect)
                data.append(datastring)
    return data, labels


# print(read_data_with_labels('data_mixed_labeled_preprocessed.tsv'))
