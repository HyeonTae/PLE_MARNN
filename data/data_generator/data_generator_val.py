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

with open(args.data_name+"_target_vocab.json", "r") as json_file:
    target_vocab = json.load(json_file)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))
os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))
data_path = "data/network_inputs/iitk-"+args.data_name+"-1189/"
data = np.load(data_path+'tokenized-examples.npy').item()
val = open(data_path+"test_val.txt", 'w')

for k in tqdm(data['validation']):
    for i in data['validation'][k]:
        #source sequence
        if i[1] == '-1':
            source = i[0]
            lines = source.count('~')
            for l in range(lines):
                if l >= 10:
                    source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
                else:
                    source = source.replace(str(l) + " ~ ", "", 1)
            source = source.replace("  ", " ")
            val.write("%s\n" % (source))

val.close()
