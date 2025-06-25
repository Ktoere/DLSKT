# @File    : dataloader.py
# @Software: PyCharm


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import itertools
from torch.utils.data import Dataset
import pandas as pd
import ast


def getData(path,max_seq_length):
    exercise_seq = []
    concept_seq = []
    response_seq = []
    attemptCount_seq = []
    hintCount_seq = []
    taken_time_seq = []
    interval_time_seq = []
    df = pd.read_csv(path)
    df['problem_id'] = df['problem_id'].apply(ast.literal_eval)
    df['skill_id'] = df['skill_id'].apply(ast.literal_eval)
    df['correct'] = df['correct'].apply(ast.literal_eval)
    df['attemptCount'] = df['attemptCount'].apply(ast.literal_eval)
    df['hintCount'] = df['hintCount'].apply(ast.literal_eval)
    df['timeTaken'] = df['timeTaken'].apply(ast.literal_eval)
    df['intervaltime'] = df['intervaltime'].apply(ast.literal_eval)

    for index, row in tqdm(df.iterrows(), total=len(df), desc='loading data', unit="row"):
        length = int(row[8])
        if length < 5:
            continue

        # row[1] = convert_list_str_to_np_array(row[1])
        segments1, segments2, segments3,segments4,segments5,segments6,segments7 = process_array(row[1], row[2], row[3], row[4],row[5], row[6],row[7],max_seq_length)
        exercise_seq.append(segments1)
        concept_seq.append(segments2)
        response_seq.append(segments3)
        attemptCount_seq.append(segments4)
        hintCount_seq.append(segments5)
        taken_time_seq.append(segments6)
        interval_time_seq.append(segments7)

    # np.vstack [[2, 5, 7], [[2, 6, 7], [34, 7, 6]]] -> [[2, 5, 7], [2, 6, 7], [34, 7, 6]]
    return np.vstack(exercise_seq), np.vstack(concept_seq), np.vstack(response_seq),np.vstack(attemptCount_seq),np.vstack(hintCount_seq),np.vstack(taken_time_seq),np.vstack(interval_time_seq)


def process_array(exer_seq, conc_seq, resp_seq,attemptCount_seq,hintCount_seq,taken_time_seq,interval_time_seq, max_length):
    """
    Process a NumPy array to ensure each segment has length max_length.
    Pad with zeros if length < max_length and split into segments if length > max_length.
    """
    # Ensure arr is a NumPy array
    exer_seq = np.asarray(exer_seq)
    conc_seq = np.asarray(conc_seq)
    resp_seq = np.asarray(resp_seq)
    attemptCount_seq = np.asarray(attemptCount_seq)
    hintCount_seq = np.asarray(hintCount_seq)

    taken_time_seq = np.asarray(taken_time_seq)
    interval_time_seq = np.asarray(interval_time_seq)
    interval_time_seq = interval_time_seq + 1


    # Case 1: Length < max_length
    if len(exer_seq) < max_length:
        # Pad with zeros at the front
        # exer_seq_padded_arr = np.pad(exer_seq, (max_length - len(exer_seq), 0), mode='constant')
        # conc_seq_padded_arr = np.pad(conc_seq, (max_length - len(conc_seq), 0), mode='constant')
        # resp_seq_padded_arr = np.pad(resp_seq, (max_length - len(resp_seq), 0), mode='constant')
        # taken_time_seq_padded_arr = np.pad(taken_time_seq, (max_length - len(taken_time_seq), 0), mode='constant')
        # interval_time_seq_padded_arr = np.pad(interval_time_seq, (max_length - len(interval_time_seq), 0), mode='constant')


        # # Pad with zeros
        exer_seq_padded_arr = np.pad(exer_seq, (0, max_length - len(exer_seq)), mode='constant')
        conc_seq_padded_arr = np.pad(conc_seq, (0, max_length - len(conc_seq)), mode='constant')
        resp_seq_padded_arr = np.pad(resp_seq, (0, max_length - len(resp_seq)), mode='constant')
        attemptCount_seq_padded_arr = np.pad(attemptCount_seq, ( 0,max_length - len(attemptCount_seq)), mode='constant')
        hintCount_seq_padded_arr = np.pad(hintCount_seq, ( 0,max_length - len(hintCount_seq)),
                                              mode='constant')

        taken_time_seq_padded_arr = np.pad(taken_time_seq, (0, max_length - len(taken_time_seq)), mode='constant')
        interval_time_seq_padded_arr = np.pad(interval_time_seq, (0, max_length - len(interval_time_seq)),
                                              mode='constant')


        # interval_time_seq_padded_arr[0] = 1
        # interval_time_seq_padded_arr[len(exer_seq) - 1] = 1441
        # taken_time_seq_padded_arr[len(exer_seq) - 1] = 301

        return np.array([exer_seq_padded_arr]), np.array([conc_seq_padded_arr]), np.array([resp_seq_padded_arr]), np.array([attemptCount_seq_padded_arr]), np.array([hintCount_seq_padded_arr]),np.array([taken_time_seq_padded_arr]),np.array([interval_time_seq_padded_arr])

    # Case 2: Length > max_length
    else:
        # Calculate the number of segments
        num_segments = int(np.ceil(len(exer_seq) / max_length))

        # Create a list to hold the segments
        exer_seq_segments = []
        conc_seq_segments = []
        resp_seq_segments = []
        attemptCount_seq_segments = []
        hintCount_seq_segments = []
        taken_time_seq_segments = []
        interval_time_seq_segments = []

        for i in range(num_segments):
            start = i * max_length
            end = min((i + 1) * max_length, len(exer_seq))
            exer_seq_segment = exer_seq[start:end]
            conc_seq_segment = conc_seq[start:end]
            resp_seq_segment = resp_seq[start:end]
            attemptCount_seq_segment = attemptCount_seq[start:end]
            hintCount_seq_segment = hintCount_seq[start:end]
            taken_time_seq_segment = taken_time_seq[start:end]
            interval_time_seq_segment = interval_time_seq[start:end]
            # Pad the last segment if necessary
            if len(exer_seq_segment) < max_length:
                # left pad with zeros
                # exer_seq_segment = np.pad(exer_seq_segment, (max_length - len(exer_seq_segment), 0), mode='constant')
                # conc_seq_segment = np.pad(conc_seq_segment, (max_length - len(conc_seq_segment), 0), mode='constant')
                # resp_seq_segment = np.pad(resp_seq_segment, (max_length - len(resp_seq_segment), 0), mode='constant')
                # taken_time_seq_segment = np.pad(taken_time_seq_segment, (max_length - len(taken_time_seq_segment), 0), mode='constant')
                # interval_time_seq_segment = np.pad(interval_time_seq_segment, (max_length - len(interval_time_seq_segment), 0), mode='constant')
                # 后补0
                exer_seq_segment = np.pad(exer_seq_segment, (0, max_length - len(exer_seq_segment)),
                                          mode='constant')
                conc_seq_segment = np.pad(conc_seq_segment, (0, max_length - len(conc_seq_segment)),
                                          mode='constant')
                resp_seq_segment = np.pad(resp_seq_segment, (0, max_length - len(resp_seq_segment)),
                                          mode='constant')

                attemptCount_seq_segment = np.pad(attemptCount_seq_segment, (0, max_length - len(attemptCount_seq_segment)),
                                          mode='constant')

                hintCount_seq_segment = np.pad(hintCount_seq_segment,
                                                  (0, max_length - len(hintCount_seq_segment)),
                                                  mode='constant')

                taken_time_seq_segment = np.pad(taken_time_seq_segment, ( 0,max_length - len(taken_time_seq_segment)),
                                                mode='constant')
                interval_time_seq_segment = np.pad(interval_time_seq_segment,
                                                   ( 0,max_length - len(interval_time_seq_segment)), mode='constant')

            # interval_time_seq_segment[0] = 1
            # interval_time_seq[len(exer_seq_segment) - 1] = 1441
            # taken_time_seq_segment[len(exer_seq_segment) - 1] = 301


            exer_seq_segments.append(exer_seq_segment)
            conc_seq_segments.append(conc_seq_segment)
            resp_seq_segments.append(resp_seq_segment)
            attemptCount_seq_segments.append(attemptCount_seq_segment)
            hintCount_seq_segments.append(hintCount_seq_segment)
            taken_time_seq_segments.append(taken_time_seq_segment)
            interval_time_seq_segments.append(interval_time_seq_segment)

        return exer_seq_segments, conc_seq_segments, resp_seq_segments,attemptCount_seq_segments,hintCount_seq_segments,taken_time_seq_segments,interval_time_seq_segments







class Data_set(Dataset):
    def __init__(self,path,max_seq_length):
        super(Data_set, self).__init__()
        self.path = path
        self.max_seq_length = max_seq_length
        self.exercise_seq, self.concept_seq, self.response_seq,self.attemptCount_seq,self.hintCount_seq,self.taken_time_seq,self.interval_time_seq = getData(self.path,self.max_seq_length)



    def __getitem__(self, index):

        return self.exercise_seq[index],self.concept_seq[index],self.response_seq[index],self.attemptCount_seq[index],self.hintCount_seq[index],self.taken_time_seq[index],self.interval_time_seq[index]


    def __len__(self):
        return len(self.exercise_seq)





if __name__ == "__main__":
    path = r"D:\Code\KT\MyKT\dataset\Processed_data\ASSIST2009\train_data.csv"
    # exercise_seq, concept_seq, response_seq = getData(path,100)
    # print(exercise_seq)
    # print("#######")

