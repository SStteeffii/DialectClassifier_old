import csv

with open('Delimiter_NewLine.txt', 'r', encoding='utf-8') as file:
    my_delimiter_newline = file.read()

with open('Delimiter_Tab.txt', 'r', encoding='utf-8') as file:
    my_delimiter_tab = file.read()

files = [['data_de_cleaned_resized(used for Translations,Tab-seperated).tsv'], ['data_de_cleaned_resized.tsv']]

for file in files:
    filename = file[0]
    tab_count = 0
    with open(filename, 'r', encoding='utf-16') as infile:
        for line in infile:
            tab_count += line.count(my_delimiter_tab)
        print(filename + ", Tab-Stopps " + str(tab_count + 1))
    with open(filename, 'r', encoding='utf-16') as infile:
        line_count = len(infile.readlines())
        print(filename + ", Lines: " + str(line_count))




