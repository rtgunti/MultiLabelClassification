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

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(Y.Eurovoc_Label)
Y = multilabel_binarizer.transform(Y.Eurovoc_Label)

Text = pd.merge(_stem_text_df, _label_df, on='Filename')
Text = Text.drop(columns = "Filename")
Text = Text.drop(columns = "Eurovoc_Label")


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=50000, lower=True)
tokenizer.fit_on_texts(Text["Text"])
sequences = tokenizer.texts_to_sequences(Text.Text)
x = pad_sequences(sequences, maxlen=400)

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import hamming_loss
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras import optimizers
x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(x, Y, test_size=0.2, random_state=71)


import keras.backend as K

def mean_pred(y_true, y_pred):
  hamming_loss(y_true,y_pred)
  return K.hamming_loss(y_true,y_pred)
model = Sequential()
model.add(Embedding(50000,64 ,input_length=400))

model.add(Conv1D(64, 5, padding='valid',activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(128, 5, padding = 'valid',activation='relu'))
model.add(Dropout(.5))
model.add(Conv1D(256, 5, padding= 'valid', activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3926, activation='sigmoid'))



adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                           epsilon=None, decay=0.0,
                           amsgrad=False)
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
model.summary()
history = model.fit(x_train_tfidf, y_train_tfidf,
                    class_weight='balanced',
                    epochs=20,
                    batch_size=32)
					
#using softmax function

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import hamming_loss
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras import optimizers
x_train_tfidf, x_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(x, Y, test_size=0.2, random_state=71)


import keras.backend as K

def mean_pred(y_true, y_pred):
  hamming_loss(y_true,y_pred)
  return K.hamming_loss(y_true,y_pred)
model = Sequential()
model.add(Embedding(50000,64 ,input_length=400))

model.add(Conv1D(64, 5, padding='valid',activation='relu'))
model.add(MaxPooling1D())
model.add(Conv1D(128, 5, padding = 'valid',activation='relu'))
model.add(Dropout(.5))
model.add(Conv1D(256, 5, padding= 'valid', activation='relu'))
model.add(MaxPooling1D())
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3926, activation='softmax'))



adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                           epsilon=None, decay=0.0,
                           amsgrad=False)
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
model.summary()
history = model.fit(x_train_tfidf, y_train_tfidf,
                    class_weight='balanced',
                    epochs=20,
                    batch_size=32)
					
model.evaluate(x_test_tfidf,y_test_tfidf)
# #predicting
y_pred = model.predict(x_test_tfidf)


# Setting the top K values to 0

data_itr = 0
temp = 0
for prob_of_element in y_pred:
  k = 0
  counter = 0
  max_elements = prob_of_element.argsort()[::-1][:6]
#   print(max_elements)
  for elements in prob_of_element:
      
      if(k == max_elements[0]):
        y_pred[data_itr][k] = 1
        
      elif(k == max_elements[1]):
      
        y_pred[data_itr][k] = 1
      elif(k == max_elements[2]):
        
        y_pred[data_itr][k] = 1
      elif(k == max_elements[3]):
    

        y_pred[data_itr][k] = 1
      elif(k == max_elements[4]):
        y_pred[data_itr][k] = 1
        
      elif(k == max_elements[5]):
        y_pred[data_itr][k] = 1
     

        
#       elif(k == max_elements[6]):
        

#         y_pred[data_itr][k] = 1
      else:
        y_pred[data_itr][k] = 0
      k =k+1
  
  data_itr = data_itr+1
  if(data_itr%1000==1):
    print(data_itr)
  
  

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
print("Hamming Loss Hammingloss",hamming_loss(y_test_tfidf,y_pred))
print("f1_score Micro Balanced ",f1_score(y_test_tfidf, y_pred,average='micro'))
print("f1_score Macro Balanced",f1_score(y_test_tfidf,y_pred,average='macro'))
jaccard_similarity_score(y_test_tfidf,y_pred)