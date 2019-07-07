# 1 
# Importing all the required libraries 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import Normalizer

# Read the Stemmed Text
_stem_text_df = pd.read_csv('../data/Stem_Text.csv')
# Remove the files without any Eglish text (feature vector)
_stem_text_df["English"] = _stem_text_df["Text"].str.contains('english version document avail sinc includ english special edit', regex=True)
#_stem_text_df['English'].value_counts()

# Read the labels into a data frame
_label_df = pd.read_csv("../data/Eurovoc_Labels.csv")

# Pivot the labels so that we have them in the required format
_label_df['pivotcol'] = 1
_lbl_trnsps_df = pd.pivot_table(_label_df,values='pivotcol',index=['Filename'],columns=['Eurovoc_Label'],aggfunc=np.sum, fill_value=0)
_label_df = _label_df.groupby(['Filename'])['Eurovoc_Label'].apply(list).reset_index()

# merge the features and labels transposed 
df_for_stemming = _stem_text_df[_stem_text_df['English']==False]
_stem_text_df = df_for_stemming.drop(columns='English')
_array_df = pd.merge(_stem_text_df, _lbl_trnsps_df, on='Filename')
_array_df = _array_df.drop(columns='Filename')

# Just get the labels
Y_labels = pd.merge(_stem_text_df, _label_df, on='Filename')
Y_labels = Y_labels.drop(columns = ["Text","Filename"])


# Code to build the Correlation matrix. This takes a lot of time. So, run once. and save
#corr = _array_df.drop(columns = "Text")
# checking the correlation between the labels
#corr = corr.corr()
#import pickle
#filename = 'corr'
#pickle.dump(corr, open(filename, 'wb'))

# Read the correlation matrix
corr = pd.read_csv("../data/Corr")
corr = corr.drop(columns = "Unnamed: 0")

# Drop the correlated columns

col = corr.columns
cols_to_drop = []
k =0
num_classes = 0
for i in range(0,len(corr)):
  for j in range(0,len(corr)):
   # try: 
      if((float(corr[col[i]][j])>.9) and  (col[j] != i) and i<j):
        if(sum(_array_df[col[i]])) < sum(_array_df[col[j]]):
          cols_to_drop.append(col[i])
        else:
          cols_to_drop.append(col[j])
        
      #print(corr[i][j])
      #print("feture1",i,"feature 2",col[j])
        num_classes = num_classes +1

# gathered the columns above. Now drop
_array_df = _array_df.drop(columns = cols_to_drop)


### List drops

for j in cols_to_drop:
  for i in Y_labels["Eurovoc_Label"]:
  
    if(j in i):
      #print(i)
      #print("removing",j)
      i.remove(j)
      #print(i)
Y = Y_labels

import gc
gc.collect()

# Garbage collect unnecessary and unrequired variables

# Initialize the model
from sklearn.preprocessing import MultiLabelBinarizer
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(Y.Eurovoc_Label)
Y = multilabel_binarizer.transform(Y.Eurovoc_Label)

# import the sparse matrix. 
import scipy.sparse as sparse
X_tfidf = sparse.load_npz('sparse_matrix.npz')

# Split the data into training and testing set. 80:20 ratio
from sklearn.model_selection import train_test_split
x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, Y, test_size=0.2, random_state=71)
# Garbage collect again
gc.collect()

#print (Y.shape)
#print (X_tfidf.shape)

# Run the model now
# play around with alpha=1e-3. 5e-4. max_iter=5,10,50? class_weight=balanced / unbalanced. 
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(penalty = "l2",loss='hinge', alpha=1e-4, random_state=42, max_iter=10, tol=None,class_weight="balanced")
clf = OneVsRestClassifier(sgd)
model = clf.fit(x_train_tfidf, y_train_tfidf)

#predicting
y_pred = clf.predict(x_test_tfidf)
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
print("Hamming Loss Hammingloss",hamming_loss(y_test_tfidf,y_pred))
print("f1_score Micro Balanced ",f1_score(y_test_tfidf,y_pred,average='micro'))
print("f1_score Macro Balanced",f1_score(y_test_tfidf,y_pred,average='macro'))
print("f1_score WEighted Balanced ",f1_score(y_test_tfidf,y_pred,average='weighted'))
print("Jaccard score ",jaccard_similarity_score(y_test_tfidf,y_pred))