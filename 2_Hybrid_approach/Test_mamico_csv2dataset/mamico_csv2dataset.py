import csv
import numpy as np

# myData = []

text = open("examplary_mamico_data.csv", "r")
text = ''.join([i for i in text]) \
    .replace(",", ";")
x = open("clean_mamico_data.csv", "w")
x.writelines(text)
x.close()
dummy = np.zeros((5, 3, 10, 3, 3))

with open('clean_mamico_data.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=';')

    for row in csv_reader:
        a = row
        if(len(a) > 7):
            dummy[int(a[0])-1, 0, int(a[1])-1, int(a[2])
                  - 1, int(a[3])-1] = float(a[4])
            dummy[int(a[0])-1, 1, int(a[1])-1, int(a[2])
                  - 1, int(a[3])-1] = float(a[5])
            dummy[int(a[0])-1, 2, int(a[1])-1, int(a[2])
                  - 1, int(a[3])-1] = float(a[6])
