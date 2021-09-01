'''
Written by: Jay Seabrum
First Created: Aug 31, 2021
Last Updated: Sep 1, 2021

Python version: 3.8.5
OS: Windows 10
'''

# Deflating the number of entries in BERT. 

# Import packages
import os as os
import re as re
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

# Set working directory
os.chdir("C:/Users/Jay Seabrum/Desktop")

# Import BERT
f = open("BERT-vocab.txt", "r", encoding="utf-8")
bert = f.read()
f.close

# Delimit
bert = bert.split("\n")

# Create an array to capture how bert's size changes over time
bert_sizes = []
bert_start_size = len(bert)
bert_sizes.append(bert_start_size)

# Last entry in bert is the empty character string. Removing.
bert = bert[0:(bert_start_size - 1)] 

# Loads of initial entries that contain a [. Removing
brackets = []
for word in bert:
    if re.search("\[", word):
        brackets.append(word)

bert = list(set(bert) - set(brackets))   
bert_sizes.append(len(bert))  

# Loads of hash entries with a preceeding ##. Removing
hashes = []
for word in bert:
    if re.search("##", word):
        hashes.append(word)

bert = list(set(bert) - set(hashes))
bert_sizes.append(len(bert))

# Numerics [0-9], and entries that contain numerics are not words. Removing
numerics = []
for word in bert:
    if re.search("[0-9]", word):
        numerics.append(word)

bert = list(set(bert) - set(numerics))
bert_sizes.append(len(bert))        



# Special characters are hard to categorize neatly
# Instead, keep entries that are letters only. 
bert_letters = []
for word in bert:
    if re.search("([a-z]|[A-Z])", word):
        bert_letters.append(word)

bert = bert_letters.copy()
bert_sizes.append(len(bert))

# Time to lemmatize with the wordnetlemmatizer
wn = WordNetLemmatizer()

# Verb lemmatizing
verb_lemma = []
verb_lemma_list = []

for word in bert:
    verb_lemma = wn.lemmatize(word, pos="v")
    verb_lemma_list.append(verb_lemma)

bert = list(set(verb_lemma_list))
bert_sizes.append(len(bert))

# Lemmatize nouns
noun_lemma = []
noun_lemma_list = []

for word in bert:
    noun_lemma = wn.lemmatize(word, pos="n")
    noun_lemma_list.append(noun_lemma)

bert = list(set(noun_lemma_list))
bert_sizes.append(len(bert))

# Lemmatize adjectives
adj_lemma = []
adj_lemma_list = []

for word in bert:
    adj_lemma = wn.lemmatize(word, pos="a")
    adj_lemma_list.append(adj_lemma)

bert = list(set(adj_lemma_list))    
bert_sizes.append(len(bert))

# Get rid of any remaining special characters
specials = []
for word in bert:
    if re.search("([^a-z]|[^A-z])", word):
        specials.append(word)

bert = list(set(bert) - set(specials))
bert_sizes.append(len(bert))

# remove entries that contain a single character
bert = [i for i in bert if len(i) > 1]

# Add back in a and i
bert.append("a")
bert.append("i")
bert_sizes.append(len(bert))

# Use the spellchecker to get rid of remaining non-words
spell = SpellChecker()

# List comprehension to get the words that are valid in the
# pyspellchecker package. 
# Documentation here:
# https://readthedocs.org/projects/pyspellchecker/downloads/pdf/latest/
cleaner = [word for word in bert if word in spell]

bert = cleaner.copy()
bert_sizes.append(len(bert))

# Save the bert object to a text file
del(f)
f = open("bert_cleaner.txt", "w")
f.write('\n'.join(bert))
f.close()

# Get ratio of the sizes
bert_sizes
ratios = [ratio / bert_sizes[0] for ratio in bert_sizes]
reduction = 1 - ratios[len(ratios) - 1]