# %%
import os
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import random

# %%
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# %%


class DontPatronizeMe:

    def __init__(self, train_path, test_path):

        self.train_path = train_path
        self.test_path = test_path
        self.train_task1_df = None
        self.train_task2_df = None
        self.test_set = None

    def load_task1(self):
        """
        Load task 1 training set and convert the tags into binary labels. 
        Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
        Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
        It returns a pandas dataframe with paragraphs and labels.
        """
        rows=[]
        with open(os.path.join(self.train_path, 'dontpatronizeme_pcl.tsv')) as f:
            for line in f.readlines()[4:]:
                par_id=line.strip().split('\t')[0]
                art_id = line.strip().split('\t')[1]
                keyword=line.strip().split('\t')[2]
                country=line.strip().split('\t')[3]
                t=line.strip().split('\t')[4].lower()
                l=line.strip().split('\t')[-1]
                if l=='0' or l=='1':
                    lbin=0
                else:
                    lbin=1
                rows.append(
                    {'par_id':par_id,
                    'art_id':art_id,
                    'keyword':keyword,
                    'country':country,
                    'text':t, 
                    'label':lbin, 
                    'orig_label':l
                    }
                    )
                
        df=pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label']) 
        self.train_task1_df = df
        return self.train_task1_df


    def load_test(self):
        #self.test_df = [line.strip() for line in open(self.test_path)]
        rows=[]
        with open(self.test_path) as f:
            for line in f.readlines()[4:]:
                t=line.strip().split('\t')[3].lower()
                rows.append(t)
        self.test_set = rows



dpm = DontPatronizeMe(train_path, test_path)

# %%
df = dpm.load_task1()

#shuffle
df = df.sample(frac = 1)

# %%


X = df['text'].to_list()
y = df['label'].to_list()
df_len = len(df)




#X_train = X[0:8397]
#X_test = X[8397:]
#y_train = y[0:8397]
#y_test = y[8397:]

#print(len(X_train))

# %%
#downsample and upsample 
#create a function that takes the two lists, zips them together into tuples
def list_tups(X, y):
    list_tups = []
    for i in range(df_len):
        pair = (X[i], y[i])
        list_tups.append(pair)
    return list_tups

#create another function that takes the list of tuples, if they're label is a 1, we are going to duplicate it 100 times
#then we will add the tuples back into a list 

def upsample(list_tups):
    new_list = []
    for pair in list_tups:
        if pair[1] == 1:
            for i in range(2):
                new_list.append(pair)
        else:
            new_list.append(pair)
    X = []
    y = []
    for pair in new_list:
        X.append(pair[0])
        y.append(pair[-1])
        
    return X, y



def downsample(X, y):
    tups = []
    for i in range(df_len):
        pair = (X[i], y[i])
        tups.append(pair)

    new_list = []
    i = 0  
    #print(tups)
    pos_count = 0
    neg_count = 0 
    
    for pair in tups:
        if pair[-1] == 1:
            pos_count+=1
            new_list.append(pair)
            
        elif pair[-1] == 0:
            neg_count +=1
            if neg_count > 700:
                new_list.append(pair)
                
        X_out = []
        y_out = []
                
    for pair in new_list:
        X_out.append(pair[0])
        y_out.append(pair[-1])
        
    #print(pos_count, neg_count)
    return X_out, y_out
                
def downsample2(list_tups):
    new_list = []
    i = 0
    for pair in list_tups:
        if pair[1] == 1:
            new_list.append(pair)
        elif pair[1] == 0:
            if i < 4000:
                i+=1
            else: 
                new_list.append(pair)
                
        X_out = []
        y_out = []
                
    for pair in new_list:
        X_out.append(pair[0])
        y_out.append(pair[1])
        
    return X_out, y_out
    

    
    
def shuffle(X, y):
    tups = []
    for i in range(df_len):
        pair = (X[i], y[i])
        tups.append(pair)
    random.shuffle(tups)
    
    X_out = []
    y_out = []
                
    for pair in tups:
        X_out.append(pair[0])
        y_out.append(pair[-1])
        
    return X_out, y_out
    
    
list_tups = list_tups(X, y)
X, y = upsample(list_tups)
#X, y = downsample(X, y)
#X, y = downsample2(list_tups)
X, y = shuffle(X, y)



# %%
def count_pos_sample(y):
    count_pos_sample = 0
    for item in y:
        if item == 1:
            count_pos_sample +=1
            
    return count_pos_sample

pos_count = count_pos_sample(y)
print(pos_count)

# %%
def count_neg_sample(y):
    count_neg_sample = 0
    for item in y:
        if item == 0:
            count_neg_sample +=1
            
    return count_neg_sample

neg_count = count_neg_sample(y)
print(neg_count)

# %%
#total and proportions:

total = neg_count + pos_count
neg_prop = float(neg_count / total)
pos_prop = float(pos_count/total)

print("negative propotion: ", neg_prop)
print("positive proportion: ", pos_prop)

# %%
vectorizer = CountVectorizer()

# %%
X = vectorizer.fit_transform(X)


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



# %%
clf = MLPClassifier(activation = 'relu', random_state=1, max_iter = 300).fit(X_train, y_train)




# %%
y_pred = clf.predict(X_test)

# %%
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# %%
prec

# %%
rec

# %%
f1

# %%
