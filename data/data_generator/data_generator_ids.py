import numpy as np
from tqdm import tqdm
import json
import os
import sys

with open("ids_target_vocab.json", "r") as json_file:
    target_vocab = json.load(json_file)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))
os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))
data_path = "data/network_inputs/iitk-ids-1189/"
data = np.load(data_path+'tokenized-examples.npy').item()
train = open(data_path+"data_train.txt", 'w')
val = open(data_path+"data_val.txt", 'w')

for k in tqdm(data['train']):
    for i in data['train'][k]:
        #source sequence
        source = i[0]
        lines = source.count('~')
        for l in range(lines):
            if l >= 10:
                source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
            else:
                source = source.replace(str(l) + " ~ ", "", 1)

        #target sequence
        if i[1] == '-1':
            fixed = source
            target = list()
            for l in range(len(fixed.split())):
                target.append("0")
        else:
            inset_line = int(i[1].split(' ~ ')[0].split()[1])+1
            if inset_line >= 10:
                insert_position = list(str(inset_line))[0] + " " + list(str(inset_line))[1] + " ~"
            else:
                insert_position = str(inset_line) + " ~"
            fixed = i[0].replace(insert_position, 'insert ' + insert_position, 1)
            for l in range(lines):
                if l >= 10:
                    fixed = fixed.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
                else:
                    fixed = fixed.replace(str(l) + " ~ ", "", 1)

            source_list = source.split()
            target = list()
            for l in range(len(source_list)):
                target.append("0")

            fixed_list = fixed.split()
            insert_start_index = fixed_list.index('insert')

            insert_string = i[1].split(" ~ ")[1].split()
            for index in range(len(insert_string)):
                target.insert(insert_start_index + index, target_vocab[insert_string[index]])

        train.write("%s\t%s\n" % (source, " ".join(target)))
train.close()

for k in tqdm(data['validation']):
    for i in data['validation'][k]:
        #source sequence
        source = i[0]
        lines = source.count('~')
        for l in range(lines):
            if l >= 10:
                source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
            else:
                source = source.replace(str(l) + " ~ ", "", 1)

        #target sequence
        if i[1] == '-1':
            fixed = source
            target = list()
            for l in range(len(fixed.split())):
                target.append("0")
        else:
            inset_line = int(i[1].split(' ~ ')[0].split()[1])+1
            if inset_line >= 10:
                insert_position = list(str(inset_line))[0] + " " + list(str(inset_line))[1] + " ~"
            else:
                insert_position = str(inset_line) + " ~"
            fixed = i[0].replace(insert_position, 'insert ' + insert_position, 1)
            for l in range(lines):
                if l >= 10:
                    fixed = fixed.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
                else:
                    fixed = fixed.replace(str(l) + " ~ ", "", 1)

            source_list = source.split()
            target = list()
            for l in range(len(source_list)):
                target.append("0")

            fixed_list = fixed.split()
            insert_start_index = fixed_list.index('insert')

            insert_string = i[1].split(" ~ ")[1].split()
            for index in range(len(insert_string)):
                target.insert(insert_start_index + index, target_vocab[insert_string[index]])

        val.write("%s\t%s\n" % (source, " ".join(target)))
val.close()
