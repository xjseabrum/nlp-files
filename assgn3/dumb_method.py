
import pandas as pd
import re as re
import random as r
import math as m
import numpy as np
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

def collect_data(txt_file):
    import pandas as pd
    # Open the text file and delimit it by a single-carriage return
    with open(txt_file, encoding = "utf-8") as file:
        datafile = file.read().split("\n")
    
    # Remove all empty strings
    datafile = list(filter(lambda x: x != "", datafile))

    # Put into a pandas datafile, delimit by tab
    datafile = pd.DataFrame(datafile)
    datafile = datafile.rename(columns = {0: "temp"})
    new_frame = datafile["temp"].str.split("\t", n = 2, expand = True)

    # Assign column names and drop old name
    datafile["word_id"] = new_frame[0].astype(int)
    datafile["word"] = new_frame[1]
    datafile["tag"] = new_frame[2]
    datafile.drop(columns = "temp", inplace = True)

    # Create a temp list for incrementing the group id
    wid = list(datafile["word_id"])
    increment = -1
    temp = []
    for item in range(len(wid)):
        if wid[item] == 1:
            increment += 1
            temp.append(increment)
        else:
            temp.append(increment)

    # Assign the group id
    datafile["group_id"] = temp
    return datafile


# Set df equal to the function above
df = collect_data("S21-gene-train.txt")
tags = list(df["tag"])



def stupid_dist():
    max = 15
    rand = 100*np.random.rand(1)
    if rand > 60:
        return 0
    if rand > 40:
        return 1
    if rand > 20:
        return 2
    if rand > 10:
        return 3
    if rand > 5:
        return 4
    if rand > 2.5:
        return 5
    else:
        return np.random.randint(low = 6, high = max+1)



test = []
# i= 0
# while i < 18000:
#     t = stupid_dist()
#     test.append(t)
#     i += 1
    

# plt.hist(test, bins=range(min(test), max(test) + 1, 1))
# plt.show()


# idx1, idx2, count, flag = 0, 0, 0, False
# tag_set = []
# # pair = ()
# for i, element in enumerate(tags):
#     if element == "B":
#         idx1 = i
#         if count > 1:
#             tag_set.append((count))
#         count = 1
#         flag = True
#     if (((element == "I")) & (flag)):
#         count += 1
#         flag = True
#     if ((element == "O") & (flag)):
#         tag_set.append(count)
#         count = 1
#         flag = False
#     if (i == len(play)):
#         tag_set.append(count)
#         count = -99
#         flag = False

# Checking word frequency
# In this first pass, will get the number of
# times words appear. 
df_word = df.groupby(["word"]).size()
df["count"] = 0
df_wordtag = pd.DataFrame(df.groupby(["word", "tag"], as_index = False)["count"].agg('size'))
df_wtp = df_wordtag.pivot(index = "word", columns = "tag", values = "size")
df_wtp = df_wtp.fillna(0)
df_wtp["n"] = df_wtp["B"] + df_wtp["I"] + df_wtp["O"]
word_count = df_wtp[["n"]]
word_count["word"] = word_count.index
word_count = word_count.reset_index(drop=True)

# Left join word count to the main df.
merged = pd.merge(df, word_count, on="word")

# Resort based on group_id, word_id
merged = merged.sort_values(by=["group_id","word_id"], ascending=[True, True]).reset_index(drop = True)

# Create word2.  Word2 is the same as word, except all words that occur
# less than 10 times will be masked with <UNK>.
merged["word2"] = merged["word"]
merged["word2"] = np.where(merged["n"] < 10, "<UNK>", merged["word2"])
merged["count"] = 0

# Now let's get stats on the most frequent word-tag types
df_wordtag = pd.DataFrame(merged.groupby(["word2", "tag"], as_index = False)["count"].agg('size'))
df_wtp = df_wordtag.pivot(index = "word2", columns = "tag", values = "size")
df_wtp = df_wtp.fillna(0)

# Get row percents
res = df_wtp.div(df_wtp.sum(axis=1), axis=0)

# Let's see the top words that have a B tag more than
# 70% of the time.
b_words = res.loc[res["B"] >= 0.8].sort_values(by="B", ascending=False)
b_word_list = list(b_words.index)
b_probs = dict(zip(b_words.index, b_words["B"]))
# And the same for I.
i_words = res.loc[res["I"] >= 0.5].sort_values(by="I", ascending=False)
i_word_list = list(i_words.index)
i_probs = dict(zip(i_words.index, i_words["I"]))


# Create a function that will check the word against the dictionary
def b_find(word: str, dict: dict):
    return word in dict.keys()

# Then create a function to see if that word will be tagged with B
def b_tag(word: str, dict: dict):
    if b_find(word, dict):
        rand = np.random.random(1)[0]
        if rand <= dict[word]:
            return "B"
    return "O"

# Then call the function to randomly choose the number of I to assign
# after B
def limit_i(current, max):
    i = stupid_dist()
    while i > (max - current):
        i = stupid_dist()
    return i


# df_wtp['prop1s'] = df_wtp[1] / (df_wtp[0] + df_wtp[1])
df_wtp["n"] = df_wtp["B"] + df_wtp["I"] + df_wtp["O"]
word_count = df_wtp[["n"]]
word_count["word"] = word_count.index
word_count = word_count.reset_index(drop=True)

play_df = merged.loc[((merged["group_id"] == 5560) | (merged["group_id"] == 5561))]
play_df["group_id"] = play_df["group_id"] - 5560

# def predict(dataframe, group_id, word_id, word_col, pred_col, dict):
#     for sent in range(max(dataframe[group_id])):
#         max_word = max(dataframe.loc[dataframe[group_id] == sent, word_id])
#         for word in range(1, max_word+1):
#             w = list(dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), word_col])[0]
#             w_tag = list(dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), pred_col])[0]
#             if w_tag == "O":
#                 p_tag = b_tag(w, dict)
#                 if p_tag == "B":
#                     dataframe.loc[((dataframe[word_id] == word) & (dataframe[group_id] == sent)), pred_col] = "B" 
#                     num_i = limit_i(word, max_word)
#                     if num_i == 0: pass
#                     dataframe.loc[((dataframe[word_id].isin(np.arange(word+1, word+num_i+1))) & (dataframe[group_id] == sent)), pred_col] = "I" 
#             else: pass
from tqdm import tqdm

merge2 = merged.copy()
merge2["pred"] = "O"
preds = []
def predict(dataframe, group_id, word_col):
    for sent in tqdm(range(max(dataframe[group_id])+1)):
        word_list = list(dataframe.loc[dataframe[group_id] == sent, word_col])
        max_words = len(word_list)
        pred_list = ["O"] * max_words
        out_list = [b_tag(x, b_probs) for x in word_list]
        b_idx = [y for y, x in enumerate(out_list) if x == "B"]
        if len(b_idx) != 0:
            num_b = len(b_idx)
            for item in range(num_b):
                num_i = limit_i(b_idx[item]+1, max_words)
                if num_i > 0:
                    opd = pd.DataFrame(out_list)
                    opd.iloc[(b_idx[item] + 1):(b_idx[item] + num_i + 1)] = "I"
                    out_list = list(opd[0])
        preds.append(out_list)

# predict(play_df, "group_id", "word_id", "word2",  "pred",  b_probs)
predict(merge2, "group_id", "word2")
output = []
for element in preds:
    for tag in element:
        output.append(tag)

merge2["preds"] = output

word_tags = merge2[["word_id", "word", "tag"]]
word_pred = merge2[["word_id", "word", "preds"]]

word_tags.to_csv("actual_tags.txt", sep="\t", header = False, index = False)
word_pred.to_csv("pred_tags.txt", sep="\t", header = False, index = False)
