#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:40:30 2019

@author: pranee
"""
from bs4 import BeautifulSoup
from pathlib import Path
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk


#give the path of your eurlex directory here
path = '/Users/pranee/Downloads/eurlex_html_EN_NOT'

#fetches the directory folder from the path
basepath = Path(path)
files_in_basepath = basepath.iterdir()
#creates a dictionary of stopwords
set_StopWords=set(stopwords.words('english'))
#Initializing the stemmer
porter_stemmer=PorterStemmer()
#uncomment below code to genrate Plain_Text.csv and the generated text contains stopwords
#with open('Plain_Text.csv', 'w', encoding="latin-1") as csvfile:

#uncomment below code to generate eurovoc_stemmed text that has no stopwords
with open('eurovoc_stemmedText.csv', 'w') as csvfile:
    fieldnames = ['Filename', 'Eurovoc_text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in files_in_basepath:
        
        print(item.name)
        if item.is_file():
            f = open(item,encoding="latin-1")
            aText=f.read()
            f.close()
            #parses the html in to text
            soup=BeautifulSoup(aText,'html.parser')
            #searches for div with class name as texte
            name_box = soup.find('div', attrs={'class': 'texte'})
            if name_box is not None:
                string_data=name_box.text
                text_unTokenized=str(string_data).replace("--","")
                #creates tokes of the parsed text
                text_tokens=nltk.word_tokenize(text_unTokenized)
                stemmed_List=[]
                
                for word in text_tokens:
                    'removes the word if it matches with the word in the dictionary
                    if word.lower() not in set_StopWords:
                        stemmed_List.append(porter_stemmer.stem(word))
                    
                #uncomment this block for generating the file in filename,list of stemmed words format
                #print(stemmed_List)
                #writer.writerow({'Filename': item.name,'Eurovoc_text': stemmed_List})
                    
                #uncomment this block for generating the file in filename,string of stemmed words format
                sFinal=' '.join(stemmed_List)
                
                
                #uncomment below code to generate the csv with untokenized text without removing stop words
                #writer.writerow({'Filename': item.name,'Text': text_unTokenized.strip()})
                
                #uncomemnt below code to genrate the csv with tokenized text that doesnt contain stop words
                writer.writerow({'Filename': item.name,'Eurovoc_text': sFinal.strip()})
                
csvfile.close()
