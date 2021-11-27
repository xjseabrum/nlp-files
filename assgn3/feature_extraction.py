# This section of the code is to find out the useful features that might be 
# in the data.

# Read data and find out what the general distribution of the data are.
# This part is done in the shell

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

def confmat(frame):
    tp = frame[1][1]
    tn = frame[0][0]
    fp = frame[1][0]
    fn = frame[0][1]
    
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    prec = tp / (tp+fp)
    rec = tn / (tn + fn)
    acc = 0.5 * (tpr + tnr)
    f1 = (2 * tp) / (2*tp + fp + fn)
    print("Precision: " + str(round(prec, 4)) + "\n" + 
          "Recall: " + str(round(rec, 4)) + "\n" + 
          "bAcc: " + str(round(acc, 4)) + "\n"
          "F1: " + str(round(f1, 4)))

# Set df equal to the function above
df = collect_data("S21-gene-train.txt")
tags = list(df["tag"])



def stupid_dist(max):
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
i= 0
while i < 18000:
    t = stupid_dist(15)
    test.append(t)
    i += 1
    

plt.hist(test, bins=range(min(test), max(test) + 1, 1))
plt.show()

# rng = np.random.default_rng()
# s = rng.poisson(0.78, 18000)
# plt.hist(s, bins=range(min(s), max(s) + 1, 1))
# plt.show()

# s2 = rng.poisson(2.45, 18000)

# s2a = [m.ceil(0.25*x) for x in s2]
# s3 = s + s2a
# s3a = [m.floor(0.9*x) for x in s3]
# count, bins, ignored = plt.hist(s3a, 14, density=True)
# plt.show()


# offset = play.copy()
# offset.insert(0, play[0])

# n = range(len(play))
# tag_set = []
# count = 0
# for element in n:
#     if play[element]=="O":
#         count = 0
#     if play[element]=="B":
#         count += 1
#     if ((play[element]=="I") & (offset[element+1] != "B")):
#         count += 1
#     else:
#         count = 0
#     tag_set.append(count)


# play = ["O","O","B","I","O","B","I","B","I","I", "O", "O", "B", "O", "B", "B", "O", "B"]
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

# idx1, idx2, idx3, flag = 0,0,0,False
# tag_set = []
# for i, element in enumerate(play):
#     idx3 = i
#     if element == "B":
#         idx1 = i
#         print(idx1)
#         flag = True
#     if (((element == "O")) & (flag)):
#         idx2 = i - 1
#         tag_set.append(idx2 - idx1 + 1)
#         flag = False
#     if (((element == "B") & (play[idx3-1] == "I")) & (flag)):
#         if idx1 == i: pass
#         tag_set.append(idx2 - idx1)
#         flag = False

        




    

# Set up a binary tag variable
tags = list(df["tag"])
bin_tag = [1 if x!="O" else 0 for x in tags]
df["bin_tag"] = bin_tag

# Get some basic stats regarding the tags.

# Tot number of each tag
num_words = len(df)
num_b = df[df["tag"] == "B"].shape[0]
num_i = df[df["tag"] == "I"].shape[0]
num_o = df[df["tag"] == "O"].shape[0]
num_not_o = df[df["tag"] != "O"].shape[0]

# Store the proportions
props = [num_b / num_words, 
         num_i / num_words, 
         num_o / num_words, 
         num_not_o / num_words]

# Get a df of the words that exclude O
tag_words = df.loc[df["tag"] != "O", "word"]
len_tag = pd.DataFrame([len(x) for x in tag_words])

beg_words = df.loc[df["tag"] == "B", "word"]
len_beg = pd.DataFrame([len(x) for x in beg_words])

in_words = df.loc[df["tag"] == "I", "word"]
len_in = pd.DataFrame([len(x) for x in in_words])

# Summary stats
len_tag.describe()
len_beg.describe()
len_in.describe()

# There is no statistical difference in word length between the tag types of B and I.

# Next do the same for the O words
out_words = df.loc[df["tag"] == "O", "word"]
len_out = pd.DataFrame([len(x) for x in out_words])
len_out.describe()

# No statistical difference in words length between tagged words and out words.
# Word length is not a valuable feature. 


# Checking camel case
words = list(df["word"])

def is_camel(list):
    output = [1 if ((x != x.lower()) & (x != x.upper())) 
              else 0 for x in list]
    return output

is_camel = is_camel(words)
df["is_camel"] = is_camel
# Replace first words' case to 0
df.loc[df["word_id"] == 1, ["is_camel"]] = 0

# Does camel case matter for tagged?
out_camel = df.loc[((df["is_camel"] == 1) & (df["tag"] != "O"))]
tag_camel = df.loc[((df["is_camel"] == 1) & (df["tag"] == "O"))]

#prop camel with respect to each class type:
prop_camel = [len(out_camel) / num_o, len(tag_camel) / num_not_o]

# 1% versus 15%.  Camel Case is likely a good feature.

punc_list = '!"#$%&\'(),./:;<=>?@[\\]^_`{|}~'

# Check for all caps
def all_caps(list):
    output = [1 if ((x == x.upper()) & (x not in punc_list))
              else 0 for x in list]
    return output

all_caps = all_caps(words)
df["all_caps"] = all_caps

# Does all caps matter?
# Does camel case matter for tagged?
out_caps = df.loc[((df["all_caps"] == 1) & (df["tag"] != "O"))]
tag_caps = df.loc[((df["all_caps"] == 1) & (df["tag"] == "O"))]

#prop camel with respect to each class type:
prop_caps = [len(out_caps) / num_o, len(tag_caps) / num_not_o]

# 5% versus 90% VERY SIGNIFICANT
# All caps matters in this task.

# Checking word frequency
df_word = df.groupby(["word"]).size()

df["count"] = 0
df_wordtag = pd.DataFrame(df.groupby(["word", "bin_tag"], as_index = False)["count"].agg('size'))

df_wtp = df_wordtag.pivot(index = "word", columns = "bin_tag", values = "size")
df_wtp = df_wtp.fillna(0)
df_wtp['prop1s'] = df_wtp[1] / (df_wtp[0] + df_wtp[1])
df_wtp["n"] = df_wtp[0] + df_wtp[1]
word_count = df_wtp[["n"]]
word_count["word"] = word_count.index
word_count = word_count.reset_index(drop=True)

# Summary on the proportion of tag = 1 over the words, for words
# mentioned at least 60 times.
# df_wtp.loc[df_wtp["n"] >= 60, "prop1s"].sort_values(0, ascending = False).head(50)
word_sort = df_wtp.copy()
word_sort["word"] = word_sort.index
word_sort = word_sort[["word", "prop1s"]].reset_index(drop=True)

# Create a column with this proportion on the dataframe
merged = pd.merge(df, word_sort, on="word")
merged = merged.sort_values(by=["group_id","word_id"], ascending=[True, True]).reset_index(drop = True)
merged["word2"] = merged["word"]
merged["word2"] = np.where(merged["n"] < 10, "<UNK>", merged["word2"])
merged["count"] = 0
df_wordtag = pd.DataFrame(merged.groupby(["word2", "bin_tag"], as_index = False)["count"].agg('size'))

df_wtp = df_wordtag.pivot(index = "word2", columns = "bin_tag", values = "size")
df_wtp = df_wtp.fillna(0)
# Get row percents
res = df_wtp.div(df_wtp.sum(axis=1), axis=0)
res["word"] = res.index
res = res.reset_index(drop = True)
res2 = res[["word", "1"]]

merged["top80"] = np.where(merged["prop1s"] >= 0.8, 1, 0)
merged["top60"] = np.where(((merged["prop1s"] >= 0.6) & (merged["prop1s"] < 0.8)), 1, 0)
merged["top40"] = np.where(((merged["prop1s"] >= 0.4) & (merged["prop1s"] < 0.6)), 1, 0)
merged["top20"] = np.where(((merged["prop1s"] >= 0.2) & (merged["prop1s"] < 0.4)), 1, 0)

del(df)
df = merged.copy()
del(merged)

# Some interesting keywords to note:
# Greek words: alpha, beta, gamma, delta, kappa are classified as being part of
# a tag at least 50%+.
# The word receptor is classified 75% of the time
# Ras, PKC, p53, and insulin, IL are classified 100% of the time.

# Create a trigger word list. These are words classified as being 
# tagged at least 70% of the time and not already captured by 
# another variable

# trigger = ["alpha", "beta", "gamma", "delta", "kappa",
#            "anti", "polymerase", "receptor", "kinases", 
#            "antigen", "kinase", "CSF", "phosphatase", "c", 
#            "AP", "NF", "cyclin", "Sp1", "IL", "insulin", 
#            "p53", "PKC", "Ras"]

# def target(list):
#     output = [1 if (x in trigger) 
#               else 0 for x in list]
#     return output

# target = target(df["word"])
# df["trigger"] = target


def ase_finder(search, store):
    for word in range(len(search)):
        if (bool((re.search(r'ase\b', search[word])) or 
                (re.search(r'ases\b', search[word])))):
            store.append(1)
        else:
            store.append(0)
    return store

ase = []
ase = ase_finder(df["word"], ase)
df["ase"] = ase

# How many words are in the typical tag?



# Checking confmat for the features.
# df_tr = pd.DataFrame(df.groupby(["bin_tag", "digit"], as_index = False)["count"].agg('size'))

# df_trp = df_tr.pivot(index = "bin_tag", columns = "digit", values = "size")
# df_trp = df_trp.fillna(0)
# df_trp['prop1s'] = df_trp[1] / (df_trp[0] + df_trp[1])
# df_trp["n"] = df_trp[0] + df_trp[1]
# df_trp
# confmat(df_trp)
# df_trp.loc[df_trp["n"] >= 60, "prop1s"].sort_values(0, ascending = False).head(50)

# Going with the following features:
# is_camel, all_caps, trigger, ase
# all features are binary
df["bias"] = 1


# Create a 72/18/10 split against the original df 
# for training/validation/holdout 
# First create the holdout df
max = max(df["group_id"])
uniq_group_id = list(set(df["group_id"]))
n_holdout = m.floor(0.1 * max)
holdout_idx = r.sample(uniq_group_id, n_holdout)

holdout_df = df[df.group_id.isin(holdout_idx)]

# Now create the model_df which is all the group_ids
# not in the holdout_df 
model_df = df[~(df.group_id.isin(holdout_idx))]

# From this set, create an 80/20 split on this new frame
train_max = len(set(model_df["group_id"]))
n_train = m.floor(0.8 * train_max)
n_valid = m.floor(0.2 * train_max)

uniq_train_group_ids = list(set(model_df["group_id"]))
len_ids = len(uniq_train_group_ids)

training_ids = r.sample(uniq_train_group_ids, n_train)
valid_ids = list(set(uniq_train_group_ids).difference(set(training_ids)))
# 80/20 split lends itself to 5-fold validation
features = ["is_camel", "all_caps", "top80", "top60", "top40", "top20", "ase", "bias"]

n_feat = len(features)


train = df[df["group_id"].isin(training_ids)]
y_train = train["bin_tag"]
x_train = train[features]

valid = df[df["group_id"].isin(valid_ids)]
y_valid = valid["bin_tag"]
x_valid = valid[features]


def sigmoid(z):
    s = 1 / (1 + m.exp(-z))
    return s

def deriv_log_loss(correct_class, est_prob, features):
    dll = np.dot((est_prob - correct_class), features)
    return dll
   
def weight_update(start_weights, eta, grad):
    new_weights = start_weights - eta * grad
    return new_weights

class SGD(object):
    def __init__(self):
        self.collect_weights = []
        self.avg_weights = []

    def sgd(self, n_iter, x, y, w, eta, update):
        for j in range(len(x)):
            x_list = list(x.iloc[j])
            y_true = y[j]
            i = 0
            while (i < n_iter):
                y_hat = sigmoid(np.dot(w, x_list))
                grad = deriv_log_loss(y_true, y_hat, x_list)
                w = weight_update(w, eta, grad)
                i += 1
                if ((i+1) % m.floor(n_iter*update) == 0):
                    perc = round(100*((i+1) / n_iter),2)
                    perc2 = round(100*((j+1) / len(x)), 2)
                    print("Iteration: " + str(i+1) + " of " + str(n_iter) + 
                          " = " + str(perc) + "% for training row " + 
                          str(j+1) + " of " + str(len(x)) + " (" + str(perc2) +
                          "% complete)")
            self.collect_weights.append(w)
        
        for idx in range(len(x.iloc[0])):
            col_list = [n[idx] for n in self.collect_weights]
            avg_sum = (1 / len(x)) * sum(col_list)
            self.avg_weights.append(avg_sum)
   
# Set up SGD
x = x_train.reset_index(drop = True)
y = y_train.reset_index(drop = True)
initial_weights = [0] * n_feat
n_iter = 1000
eta = 0.1
starttime = datetime.now()

sgd = SGD()
sgd.sgd(n_iter = n_iter, 
        x = x,
        y = y,
        w = initial_weights, 
        eta = eta,
        update = 1)

endtime = datetime.now()
difftime = endtime - starttime

print("SGD runtime: ", difftime)

# Pickle the dev weights for later use.  Takes 91 min to run
# the test set.  Don't want to wait on that again.
# [7.917653315667114, 5.117136751959736, 10.540264679440895, 3.943252142021299, -9.014895739402096]
#[9.830728851064604, 6.551916215723501, 12.927666067316862, 5.169799672007105, -11.118672541297002]
# [0.2061024015176023, 2.1136916154587535, 15.801972620700258, 7.5434437523725215, 5.760138518679846, 5.692263464906676, -0.7450213202439026, -9.680296995864309]

# dev_weights = sgd.avg_weights
# dev_weights_file = open("dev_weights_file.pkl", "wb")
# pickle.dump(dev_weights, dev_weights_file)
# dev_weights_file.close()

with open("dev_weights_file.pkl", "rb") as file:
    dev_weights = pickle.load(file)

# Use these weights on the validation set

est = []
v = valid.reset_index(drop = True)
x_v = x_valid.reset_index(drop = True)
for row in range(len(x_v)):
    x_valid_row = x_v.iloc[row]
    s = sigmoid(np.dot(dev_weights, x_valid_row))
    est.append(s)


# pred_class = [(1 if x >= 0.6 else 0) for x in est]
pred_class = [int(round(x, 0)) for x in est]
y_v = y_valid.reset_index(drop = True)

v["Pred"] = pred_class
v["PredLbl"] = "O"
v.loc[v["Pred"] == 1, ["PredLbl"]] = "I"

adj_lbl = list(v["PredLbl"])
start = adj_lbl[0]
comp_lbl = adj_lbl.copy()
comp_lbl.insert(0, start)

for item in range(len(adj_lbl)):
    if ((adj_lbl[item] == "I") & (adj_lbl[item] != comp_lbl[item])):
        adj_lbl[item] = "B"
    else:
        pass

v["PredLbl"] = adj_lbl

# Save the actual tags and pred to file
# so as to run the eval script.
word_tags = v[["word_id", "word", "tag"]]
word_pred = v[["word_id", "word", "PredLbl"]]

word_tags.to_csv("actual_tags.txt", sep="\t", header = False, index = False)
word_pred.to_csv("pred_tags.txt", sep="\t", header = False, index = False)