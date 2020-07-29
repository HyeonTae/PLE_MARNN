import numpy as np
from tqdm import tqdm
import argparse

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='typo',
                 type=str, help="select a data name (ids/typo)")
args = par.parse_args()

data_path = "../network_inputs/iitk-"+args.data_name+"-1189/"
data = np.load(data_path+'testing-tokenized-examples.npy').item()
train = open(data_path+"data_train.txt", 'w')
val = open(data_path+"data_val.txt", 'w')

for k in tqdm(data['train']):
    for i in data['train'][k]:
        train.write("%s\t%s\n" % (i[0], i[5]))

train.close()

for k in tqdm(data['validation']):
    for i in data['validation'][k]:
        val.write("%s\t%s\n" % (i[0], i[5]))

val.close()
