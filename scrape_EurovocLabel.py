#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script loops through the raw HTML files of EurLex dataset and scrapes the filename and the Eurovoc Descriptors

from bs4 import BeautifulSoup
from pathlib import Path
import csv
# set the path from where the raw files need to be read
path = '../data/eurlex_html_EN_NOT' 
#path = '../data/TestData' # set the Test Data

#Fetches the directory of the path
raw_files_dir = Path(path)
files_in_basepath = raw_files_dir.iterdir()

# write the labels to an output csv file
with open('../data/Eurovoc_Labels.csv', 'w', encoding="latin-1") as csvfile:
    #sets the headers in the csv file
    fieldnames = ['Filename', 'Eurovoc_Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for item in files_in_basepath:
        if item.is_file():
            f = open(item,encoding="latin-1")
            aText=f.read()
            f.close()
            #parses html to text
            soup=BeautifulSoup(aText,'html.parser')
            #searches for the div with class name listNotice as the eurovoc descriptors are in the form of list in this div
            name_box = soup.find('div', attrs={'class': 'listNotice'})
            if name_box is not None:
                ul_String=name_box.find('ul').find_next_sibling('ul')
                li_eurovoc_String=ul_String.find('li')
                if li_eurovoc_String is None:
                    #finds the orderd unorderd list that holds eurovoc label
                    ul_String=name_box.find('ul').find_next_sibling('ul').find_next_sibling('ul')
                    li_eurovoc_String=ul_String.find('li')
            
                a_eurovoc_String=li_eurovoc_String.find_all('a')
                for child in a_eurovoc_String:
                    s_data=child.string
                    #writes the data to excel in the form of two columns where first column contains filename and the second column contains the label
                    writer.writerow({'Filename': item.name,'Eurovoc_Label': s_data.strip()})
csvfile.close()
