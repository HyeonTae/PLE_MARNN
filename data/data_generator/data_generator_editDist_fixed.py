import numpy as np
from tqdm import tqdm
import json
import os
import sys
import argparse

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='typo',
                 type=str, help="select a data name (ids/typo)")
args = par.parse_args()
op = "insert"
if args.data_name == "typo":
    op = "replace"

with open(args.data_name + "_target_vocab.json", "r") as json_file:
    target_vocab = json.load(json_file)

with open("target_vocab_reverse.json", "r") as json_file:
    inverse_vocab = json.load(json_file)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))
os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))
from util.helpers import apply_fix
data_path = "data/network_inputs/iitk-"+args.data_name+"-1189/"
data = np.load(data_path+'tokenized-examples.npy').item()
train = open(data_path+"data_train_edit.txt", 'w')
val = open(data_path+"data_val_edit.txt", 'w')

def getEditDistance(a, b):
    dist = np.zeros((len(a) + 1, len(b) + 1),dtype=np.int64)
    dist[:, 0] = list(range(len(a) + 1))
    dist[0, :] = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            insertion = dist[i, j - 1] + 1
            deletion = dist[i - 1, j] + 1
            match = dist[i - 1, j - 1]
            if a[i - 1] != b[j - 1]:
                match += 1  # -- mismatch
            dist[i, j] = min(insertion, deletion, match)
    return dist

def getTrace(a, b, dist):
    log = list()
    i, j = len(a),len(b)
    while i != 0 or j != 0:
        s = min(dist[i-1][j], dist[i-1][j-1], dist[i][j-1])
        if s == dist[i][j]:
            i -= 1
            j -= 1
        else:
            if s == dist[i-1][j]:
                log.append(["d", i-2])
                i -= 1
            elif s == dist[i][j-1]:
                log.append(["i", i-1, b[j-2]])
                j -= 1
            elif s == dist[i-1][j-1]:
                log.append(["r", i-2, a[i-2]])
                i -= 1
                j -= 1
    return log

def apply_edits(source, edits):
    fixed = []
    inserted = 0
    insert_tok = [str(i) for i in range(1,110)]
    for i, edit in enumerate(edits):
        if edit == '0':
            fixed.append(source[i - inserted])
        elif edit != '-1':
            fixed.append(inverse_vocab[edit])
            if edits[inserted] not in insert_tok:
                inserted += 1
    
    return fixed

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
        source = source.replace("  ", " ")
        source = source.split()
        #target sequence
        if i[1] == '-1':
            target = target = ["0" for i in range(len(source))]
        else:
            fixed = apply_fix(i[0], i[1], op)
            lines = fixed.count('~')
            for l in range(lines):
                if l >= 10:
                    fixed = fixed.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
                else:
                    fixed = fixed.replace(str(l) + " ~ ", "", 1)
            fixed = fixed.replace("  ", " ")
            fixed = fixed.split()

            log = getTrace(source, fixed, getEditDistance(source, fixed))
            target = ["0" for i in range(len(source))]
            for l in log:
                if l[0] == "i":
                    target.insert(l[1], target_vocab[l[2]])
                elif l[0] == "r":
                    target[l[1]] = target[l[1]].replace(target[l[1]], target_vocab[l[2]])
                elif l[0] == "d":
                    target[l[1]] = target[l[1]].replace(target[l[1]], "-1")

        train.write("%s\t%s\n" % (" ".join(source), " ".join(target)))

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
        source = source.replace("  ", " ")
        source = source.split()
        #target sequence
        if i[1] == '-1':
            target = target = ["0" for i in range(len(source))]
        else:
            fixed = apply_fix(i[0], i[1], op)
            lines = fixed.count('~')
            for l in range(lines):
                if l >= 10:
                    fixed = fixed.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
                else:
                    fixed = fixed.replace(str(l) + " ~ ", "", 1)
            fixed = fixed.replace("  ", " ")
            fixed = fixed.split()

            log = getTrace(source, fixed, getEditDistance(source, fixed))
            
            target = ["0" for i in range(len(source))]
            for l in log:
                if l[0] == "i":
                    target.insert(l[1], target_vocab[l[2]])
                elif l[0] == "r":
                    target[l[1]] = target[l[1]].replace(target[l[1]], target_vocab[l[2]])
                elif l[0] == "d":
                    target[l[1]] = target[l[1]].replace(target[l[1]], "-1")

        val.write("%s\t%s\n" % (" ".join(source), " ".join(target)))

val.close()
