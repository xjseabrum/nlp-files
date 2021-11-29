import pandas as pd
import re as re
import random as r
import math as m
import numpy as np
from datetime import datetime
import pickle

###############################################################################
# Section 0:
# Function/Class definitions.

# This section defines functions and classes that will be used later in the script.

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

# Use the collect_data function to 
# set up the initial df.
df = collect_data("S21-gene-train.txt")
tags = list(df["tag"])

# Set up a binary tag (bin_tag) variable from the BIO tags
# B/I will be 1, O will be 0
tags = list(df["tag"])
bin_tag = [1 if x!="O" else 0 for x in tags]
df["bin_tag"] = bin_tag

###############################################################################
# Section 1:
# Data exploration.
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

# Q1: Does word length matter?
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

# Q2: Does camel case matter?
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

# Prop camel with respect to each class type:
prop_camel = [len(out_camel) / num_o, len(tag_camel) / num_not_o]
# 1% versus 15%.  Camel Case is likely a good feature.


# Q3: Does all caps matter?
punc_list = '!"#$%&\'(),./:;<=>?@[\\]^_`{|}~'

# Check for all caps
def all_caps(list):
    output = [1 if ((x == x.upper()) & (x not in punc_list))
              else 0 for x in list]
    return output

all_caps = all_caps(words)
df["all_caps"] = all_caps

out_caps = df.loc[((df["all_caps"] == 1) & (df["tag"] != "O"))]
tag_caps = df.loc[((df["all_caps"] == 1) & (df["tag"] == "O"))]

# Prop caps with respect to each class type:
prop_caps = [len(out_caps) / num_o, len(tag_caps) / num_not_o]
# 5% versus 90% VERY SIGNIFICANT
# All caps matters in this task.


# Q4: What is the tag frequency per word?
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

# This object will contain word + frequency count of that word.
word_count = word_count.reset_index(drop=True)

# This object will contrain word + proportion of the time that that word is a 1.
word_sort = df_wtp.copy()
word_sort["word"] = word_sort.index
word_sort = word_sort[["word", "prop1s"]].reset_index(drop=True)

# Combine the two frames into one.
word_stats = pd.merge(word_count, word_sort, on="word")

# Create a column with this proportion & count on the dataframe
merged = pd.merge(df, word_stats, on="word")

# Sort it so that the original order is preserved
merged = merged.sort_values(by=["group_id", "word_id"], ascending=[True, True]).reset_index(drop = True)

# Next, create a column called word2 that will be the same as word
# except all words with an n < 10 will be masked with <UNK>

merged["word2"] = merged["word"]
merged["word2"] = np.where(merged["n"] < 10, "<UNK>", merged["word2"])

# Now we have to reassign prop1s for words that are now "<UNK>"
# Let's get the stats for UNK words.
merged["count"] = 0
df_unk = pd.DataFrame(merged.groupby(["word2", "bin_tag"], as_index = False)["count"].agg('size'))

df_wtp = df_unk.pivot(index = "word2", columns = "bin_tag", values = "size")
df_wtp = df_wtp.fillna(0)

# Get row percents
res = df_wtp.div(df_wtp.sum(axis=1), axis=0)
res["word2"] = res.index
res = res.reset_index(drop = True)
res2 = res[["word2", 1]]
res.loc[res["word2"] == "<UNK>"]

# According to these data, UNK is typically tagged ~21% of the time.
# This proportion will be used for words that are in the holdout
# but have never been seen by the model.
unk_prop = res.loc[res["word2"] == "<UNK>"][1]

# Zip word2 and prop1 into a dictionary for lookup later with
# the holdout set that will be provided on Nov 29, 2021.
obs_props = dict(zip(res2["word2"], res2[1]))


# For this feature, we will group words based on their observed frequency of 
# being tagged.
# We will do this against the obs_props dictionary that we have made.

merged["prop1sb"] = merged["prop1s"]
merged["prop1sb"] = np.where(merged["word2"] == "<UNK>", unk_prop, merged["prop1sb"])

# [80-100%] = top80
# [60-80%) = top60
# [40-60%) = top40
# [20-40) = top20
merged["top80"] = np.where(merged["prop1s"] >= 0.8, 1, 0)
merged["top60"] = np.where(((merged["prop1s"] >= 0.6) & (merged["prop1s"] < 0.8)), 1, 0)
merged["top40"] = np.where(((merged["prop1s"] >= 0.4) & (merged["prop1s"] < 0.6)), 1, 0)
merged["top20"] = np.where(((merged["prop1s"] >= 0.2) & (merged["prop1s"] < 0.4)), 1, 0)


# Rename the frame to df.
del(df)
df = merged.copy()
del(merged)


# Q4: Do certain endings to words matter?
# Create a function to flag words that end in 
# ase or ases.
def ase_finder(search, store):
    for word in range(len(search)):
        if (bool((re.search(r'ase\b', search[word])) or 
                (re.search(r'ases\b', search[word])))):
            store.append(1)
        else:
            store.append(0)
    return store

ase = []
ase = ase_finder(df["word2"], ase)
df["ase"] = ase


# Feature finalization.
# Going with the following features:
# is_camel, all_caps, top{80,60,40,20}, ase
# all features are binary
df["bias"] = 1

# Define the features that we will use
features = ["is_camel", "all_caps", "top80", "top60", "top40", "top20", "ase",  "bias"]
n_feat = len(features)

###############################################################################
# Section 2:
# Setting up and running the model.

# Create a "72/18/10" split against the original df 
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

# Create the train and valid dfs.
train = df[df["group_id"].isin(training_ids)]
y_train = train["bin_tag"]
x_train = train[features]

valid = df[df["group_id"].isin(valid_ids)]
y_valid = valid["bin_tag"]
x_valid = valid[features]

# Set up SGD
x = x_train.reset_index(drop = True)
y = y_train.reset_index(drop = True)
initial_weights = [0] * n_feat
n_iter = 5000
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
# [0.2061024015176023, 2.1136916154587535, 15.801972620700258, 7.5434437523725215, 5.760138518679846, 5.692263464906676, -0.7450213202439026, -9.680296995864309]
# [0.794689246437309, 0.9555046737358514, 28.45553444750359, 15.350487774377099, 11.395658680398874, 8.308560187398774, -3.705105971234466, -14.790733946748693]

dev_weights = sgd.avg_weights
dev_weights_file = open("dev_weights_file.pkl", "wb")
pickle.dump(dev_weights, dev_weights_file)
dev_weights_file.close()

# Uncomment this section when returning to work on this
# problem without having to rerun the entire SGD setup.
# with open("dev_weights_file.pkl", "rb") as file:
#     dev_weights = pickle.load(file)

###############################################################################
# Section 3:
# Validation

# Use these weights on the validation set
est = []
v = valid.reset_index(drop = True)
x_v = x_valid.reset_index(drop = True)
for row in range(len(x_v)):
    x_valid_row = x_v.iloc[row]
    s = sigmoid(np.dot(dev_weights, x_valid_row))
    est.append(s)

# Use typical 50/50 rounding to assign the pred numbers to 0 or 1.
pred_class = [int(round(x, 0)) for x in est]
y_v = y_valid.reset_index(drop = True)

# Assign I to 1 and O to 0.
v["Pred"] = pred_class
v["PredLbl"] = "O"
v.loc[v["Pred"] == 1, ["PredLbl"]] = "I"

# Create an adjusted label list and an offset (comp_lbl)
# list to reassign the first I after an O to be a B.
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
# to run the eval script.
word_tags = v[["word_id", "word", "tag"]]
word_pred = v[["word_id", "word", "PredLbl"]]

word_tags.to_csv("actual_tags.txt", sep="\t", header = False, index = False)
word_pred.to_csv("pred_tags.txt", sep="\t", header = False, index = False)

# Results using evalNER.py, 1k iter:
# Precision:  0.4432086282440175 
# Recall:  0.4241935483870968
# F1-measure:  0.4334926652381737

# Results using evalNER.py, 5k iter:
# Precision:  0.4019178082191781 
# Recall:  0.4732258064516129 
# F1-measure:  0.4346666666666667