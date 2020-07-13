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

with open("target_vocab.json", "r") as json_file:
    target_vocab = json.load(json_file)
    
with open('target_vocab_reverse.json', "r") as json_file:
    inverse_vocab = json.load(json_file)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))
os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))

from util.helpers import apply_fix, getTrace, getEditDistance, apply_edits, tokens_to_source, compilation_errors

data_path = "data/network_inputs/iitk-"+args.data_name+"-1189/"
data = np.load(data_path+'tokenized-examples.npy').item()
train = open(data_path+"data_train_edit.txt", 'w')
val = open(data_path+"data_val_edit.txt", 'w')

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
            fixed = source
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
                    target.insert(l[1], target_vocab["insert"][l[2]])
                elif l[0] == "r":
                    target[l[1]] = target[l[1]].replace(target[l[1]], target_vocab["replace"][l[2]])
                elif l[0] == "d":
                    target[l[1]] = target[l[1]].replace(target[l[1]], "-1")
                    
            assert (tokens_to_source(' '.join(fixed), inverse_vocab, False) == tokens_to_source(' '.join(apply_edits(source, target, inverse_vocab)), inverse_vocab, False))

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
            fixed = source
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
                    target.insert(l[1], target_vocab["insert"][l[2]])
                elif l[0] == "r":
                    target[l[1]] = target[l[1]].replace(target[l[1]], target_vocab["replace"][l[2]])
                elif l[0] == "d":
                    target[l[1]] = target[l[1]].replace(target[l[1]], "-1")
                    
            assert (tokens_to_source(' '.join(fixed), inverse_vocab, False) == tokens_to_source(' '.join(apply_edits(source, target, inverse_vocab)), inverse_vocab, False))

        val.write("%s\t%s\n" % (" ".join(source), " ".join(target)))

val.close()
