'''
Written by: Jay Seabrum (Xajavion Seabrum)
First Created: Aug 29, 2021
Last Updated: Aug 30, 2021
'''

# Note, this likely isn't the prettiest way to go about
# doing this but at least it gets the job done. 

# Import NLTK
import os as os
import re as re
import numpy as np
import scipy.stats as st
from random import sample
from nltk.stem import WordNetLemmatizer

# Set working directory
os.chdir("C:/Users/Jay Seabrum/Desktop")

# Bring in the PorterStemmer
wn = WordNetLemmatizer()
porter = PorterStemmer()
lancaster = LancasterStemmer()

# Bring in the word list from Ubuntu
f = open("american-english.txt", "r")
wl = f.read()
f.close()

# Delimit 
wl = wl.split("\n")

# Get rid of strange words/characters at the end of the file
wl = wl[0:102287]

# Get rid of proper nouns and names at the beginning of the file
wl = wl[19110:]

# Catch 's
appS = []
for word in wl:
    if re.search("'s", word):
        appS.append(word)

# Remove 's
wlNoApp = list(set(wl) - set(appS))

# remove ings
ings = []
for word in wlNoApp:
    if re.search(".{3}ing", word):
        ings.append(word)

wlNoIngs = list(set(wlNoApp) - set(ings))        

# Remove single letters
wlNoIngs = [i for i in wlNoIngs if len(i) > 1]

# Add back in A and I
wlNoIngs.append("a")
wlNoIngs.append("I")

# get uniques
wlU = list(set(wlNoIngs))

# Verb lemmatizing
wlV = []
wlStem = []

for word in wlU:
    wlV = wn.lemmatize(word, pos="v")
    wlStem.append(wlV)


wlUnique = list(set(wlStem))

# Lemmatize nouns
wlN = []
wlNS = []

for word in wlUnique:
    wlN = wn.lemmatize(word, pos="n")
    wlNS.append(wlN)


wlNUnique = list(set(wlNS))    


# Lemmatize Adjectives
wlJ = []
wlJS = []

for word in wlNUnique:
    wlJ = wn.lemmatize(word, pos="a")
    wlJS.append(wlJ)

wlJUnique = list(set(wlJS)) 

# Final length of the dataset
dfSize = len(wlJUnique)
dfSize

# 31098

# Use ulJUnique as the sampling space

# Randomly sample 100 words from here
one = sample(wlJUnique, 100)
two = sample(wlJUnique, 100)
three = sample(wlJUnique, 100)
four = sample(wlJUnique, 100)
five = sample(wlJUnique, 100)
six = sample(wlJUnique, 100)
seven = sample(wlJUnique, 100)
eight = sample(wlJUnique, 100)
nine = sample(wlJUnique, 100)
ten = sample(wlJUnique, 100)


# Create knowledge() to run the trials interactively
results = []
def knowledge(df):
    tot = 0
    know = 0
    for word in df:
        print(word)
        ans = input("Do you know this? ")
        if ans == "y":
            know += 1
        tot += 1
        percRight = know / tot
    results.append(percRight)
    return know, tot, percRight

# Then run, iteratively, knowledge(one), knowledge(two), etc.
# Show results after the ten trials
results
# [0.89, 0.9, 0.84, 0.89, 0.95, 0.91, 0.87, 0.94, 0.87, 0.85]

# Calculate the mean and std.
np.mean(results)
np.std(results)

# Calculate the 95% CI with scipy
percCIRange = np.round(st.t.interval(0.95, len(results) - 1, loc=np.mean(results), scale=st.sem(results)), 2)

percCIRange
# array([0.87, 0.92])


# Finally calculate the range with respect to the size of 
# the dataset

countCIRange = dfSize * percCIRange
countCIRange

# array([27055.26, 28610.16])
    






