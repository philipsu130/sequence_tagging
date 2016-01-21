"""
Simple script to split training data into training data and validation data. The original
"train.txt" is split into 2 subsets of data "train_partial.txt" and "validation.txt". Data
is split in a 1:4 ratio.
"""
with open('train.txt', 'r') as f:
    train = open('train_partial.txt', 'w')
    validation = open('validation_partial.txt', 'w')
    i = 0
    while True:
        tokens = f.readline()
        pos = f.readline()
        labels = f.readline()
        if not tokens or not pos or not labels:
            break
        i += 1
        # Split every 5th data point
        if i == 5:
            validation.writelines([tokens, pos, labels])
            i = 0
        else:
            train.writelines([tokens, pos, labels])
    train.close()
    validation.close()



