"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from util.tokenizer import EmptyProgramException
from util.helpers import get_rev_dict, make_dir_if_not_exists
from util.helpers import apply_fix, getTrace, getEditDistance, apply_edits
import os
import time
import argparse
import sqlite3
import numpy as np
from functools import partial
import json
from tqdm import tqdm

with open("data/data_generator/target_vocab.json", "r") as json_file:
    target_vocab = json.load(json_file)

class FixIDNotFoundInSource(Exception):
    pass

def remove_line_numbers(source):
    lines = source.count('~')
    for l in range(lines):
        if l >= 10:
            source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
        else:
            source = source.replace(str(l) + " ~ ", "", 1)
    source = source.replace("  ", " ")
    return source

def get_target(inp, source, fix, op):
    if fix == '-1':
        target = target = ["0" for i in range(len(inp))]
    else:
        fixed = apply_fix(source, fix, op)
        fixed = remove_line_numbers(fixed).split()
        log = getTrace(inp, fixed, getEditDistance(inp, fixed))
        target = ["0" for i in range(len(inp))]
        for l in log:
            if l[0] == "i":
                target.insert(l[1], target_vocab["insert"][l[2]])
            elif l[0] == "r":
                target[l[1]] = target[l[1]].replace(target[l[1]], target_vocab["replace"][l[2]])
            elif l[0] == "d":
                target[l[1]] = target[l[1]].replace(target[l[1]], "-1")

    return " ".join(target)

def rename_ids_(rng, corrupt_program, fix):
    corrupt_program_new = ''
    fix_new = ''

    names = []
    for token in corrupt_program.split():
        if '_<id>_' in token:
            if token not in names:
                names.append(token)

    rng.shuffle(names)
    name_dictionary = {}

    for token in corrupt_program.split():
        if '_<id>_' in token:
            if token not in name_dictionary:
                name_dictionary[token] = '_<id>_' + \
                    str(names.index(token) + 1) + '@'

    for token in fix.split():
        if '_<id>_' in token:
            if token not in name_dictionary:
                raise FixIDNotFoundInSource

    # Rename
    for token in corrupt_program.split():
        if '_<id>_' in token:
            corrupt_program_new += name_dictionary[token] + " "
        else:
            corrupt_program_new += token + " "

    for token in fix.split():
        if '_<id>_' in token:
            fix_new += name_dictionary[token] + " "
        else:
            fix_new += token + " "

    return corrupt_program_new, fix_new


def generate_training_data(db_path, bins, validation_users, min_program_length, max_program_length,
                           max_fix_length, kind_mutations, max_mutations, max_variants, seed):
    rng = np.random.RandomState(seed)

    if kind_mutations == 'typo':
        from data_processing.typo_mutator import LoopCountThresholdExceededException, FailedToMutateException, Typo_Mutate, typo_mutate
        mutator_obj = Typo_Mutate(rng)
        mutate = partial(typo_mutate, mutator_obj)
        op = "replace"

        def rename_ids(x, y): return (x, y)
    else:
        from data_processing.undeclared_mutator import LoopCountThresholdExceededException, FailedToMutateException, id_mutate
        mutate = partial(id_mutate, rng)
        rename_ids = partial(rename_ids_, rng)
        op = "insert"

    result = {'train': {}, 'validation': {}}

    exceptions_in_mutate_call = 0
    total_mutate_calls = 0
    program_lengths, fix_lengths = [], []

    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT user_id, code_id, tokenized_code, name_dict, name_seq FROM Code " +\
            "WHERE problem_id=? and codelength>? and codelength<? and errorcount=0;"
        for problem_id in tqdm(problem_list):
            for row in cursor.execute(query, (problem_id, min_program_length, max_program_length)):
                user_id, code_id, tokenized_code = map(str, row[:-2])
                name_dict, name_sequence = json.loads(
                    row[3]), json.loads(row[4])
                key = 'validation' if user_id in validation_users[problem_id] else 'train'

                program_length = len(tokenized_code.split())
                program_lengths.append(program_length)

                if program_length >= min_program_length and program_length <= max_program_length:
                    id_renamed_correct_program, _ = rename_ids(
                        tokenized_code, '')

                    # Correct pairs
                    dummy_fix_for_correct_program = '-1'
                    source = remove_line_numbers(id_renamed_correct_program)
                    target = get_target(source.split(), id_renamed_correct_program,
                            dummy_fix_for_correct_program, op)
                    try:
                        result[key][problem_id] += [
                            (source, name_dict, name_sequence,
                                user_id, code_id, target)]
                    except:
                        result[key][problem_id] = [
                            (source, name_dict, name_sequence,
                                user_id, code_id, target)]

                    # Mutate
                    total_mutate_calls += 1
                    try:
                        iterator = mutate(
                            tokenized_code, max_mutations, max_variants)

                    except FailedToMutateException:
                        exceptions_in_mutate_call += 1
                    except LoopCountThresholdExceededException:
                        exceptions_in_mutate_call += 1
                    except ValueError:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    except AssertionError:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    except Exception:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    else:
                        for corrupt_program, fix in iterator:
                            corrupt_program_length = len(
                                corrupt_program.split())
                            fix_length = len(fix.split())
                            fix_lengths.append(fix_length)

                            if corrupt_program_length >= min_program_length and \
                               corrupt_program_length <= max_program_length and fix_length <= max_fix_length:

                                try:
                                    corrupt_program, fix = rename_ids(
                                        corrupt_program, fix)
                                except FixIDNotFoundInSource:
                                    exceptions_in_mutate_call += 1

                                source = remove_line_numbers(corrupt_program)
                                target = get_target(source.split(), corrupt_program, fix, op)

                                try:
                                    result[key][problem_id] += [
                                        (source, name_dict, name_sequence, user_id, code_id, target)]
                                except:
                                    result[key][problem_id] = [
                                        (source, name_dict, name_sequence, user_id, code_id, target)]

    program_lengths = np.sort(program_lengths)
    fix_lengths = np.sort(fix_lengths)

    print('Statistics')
    print('----------')
    print('Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', program_lengths[int(0.95 * len(program_lengths))])
    try:
        print('Mean fix length: Mean =', np.mean(fix_lengths), '\t95th %ile = ', fix_lengths[int(0.95 * len(fix_lengths))])
    except Exception as e:
        print(e)
        print('fix_lengths')
        print(fix_lengths)
    print('Total mutate calls:', total_mutate_calls)
    print('Exceptions in mutate() call:', exceptions_in_mutate_call, '\n')

    return result

######## data generation #########

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Process 'C' dataset to be used in repair tasks")
    parser.add_argument(
        "-i", "--ids", help="Generate inputs for undeclared-ids-neural-network", action="store_true")
    args = parser.parse_args()

    kind_mutations = 'ids' if args.ids else 'typo'

    drop_ids = kind_mutations == 'typo'
    max_program_length = 450
    min_program_length = 75
    max_fix_length = 25

    max_mutations = 5
    max_variants = 4 if kind_mutations == 'ids' else 2

    db_path = os.path.join('data', 'iitk-dataset', 'dataset.db')
    validation_users = np.load(os.path.join('data', 'iitk-dataset', 'validation_users.npy')).item()
    bins = np.load(os.path.join('data', 'iitk-dataset', 'bins.npy'))

    seed = 1189

    output_directory = os.path.join('data/network_inputs', 'iitk-%s-%d' % (kind_mutations, seed))

    print('output_directory:', output_directory)
    make_dir_if_not_exists(os.path.join(output_directory))

    result = generate_training_data(db_path, bins, validation_users,
                                    min_program_length, max_program_length, max_fix_length,
                                    kind_mutations, max_mutations, max_variants, seed)

    np.save(os.path.join(output_directory, 'testing-tokenized-examples.npy'), result)
    print('\n\n--------------- all outputs written to {} ---------------\n\n'.format(output_directory))
