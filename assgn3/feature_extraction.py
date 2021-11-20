# This section of the code is to find out the useful features that might be 
# in the data.

# Read data and find out what the general distribution of the data are.
# This part is done in the shell

import pandas as pd
import numpy as np
import re as re

def collect_data(txt_file) -> list:
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

# Summary on the proportion of tag = 1 over the words, for words
# mentioned at least 60 times.

df_wtp.loc[df_wtp["n"] >= 60, "prop1s"].sort_values(0, ascending = False).head(40)

# Some interesting keywords to note:
# Greek words: alpha, beta, gamma, delta, kappa are classified as being part of
# a tag at least 50%+.
# The word receptor is classified 75% of the time
# Ras, PKC, p53, and insulin, IL are classified 100% of the time.

# Create a trigger word list. These are words classified as being 
# tagged at least 70% of the time and not already captured by 
# another variable

trigger = ["alpha", "beta", "gamma", "delta", "kappa",
           "anti", "polymerase", "receptor", "kinases", 
           "antigen", "kinase", "CSF", "phosphatase", "c", 
           "AP", "NF", "cyclin", "Sp1", "IL", "insulin", 
           "p53", "PKC", "Ras"]

# Create a flag for capturing "ase", or "ases" in the word.
ase = ["ase", "ases"]

def target(list):
    output = [1 if (x in trigger) 
              else 0 for x in list]
    return output

target = target(df["word"])
df["trigger"] = target


def ase_finder(search, store):
    import re as re
    for word in range(len(search)):
        if (bool((re.search(r'$ase', search[word])) or 
                (re.search(r'$ases', search[word])))):
            store.append(1)
        else:
            store.append(0)
