#imports
import re
import os
import string
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filename','-f', help="Enter Filename",type=str)
args=parser.parse_args()
filename=vars(args)["filename"]+".txt"


#set path for file
Data_Path="{}".format(filename)
Write_Path="Clean/{}".format(filename)

#clean file
def cleaner(file_path):
    text = open(file_path).read()
    text=re.sub('[¬é\xadßù]', '',text).strip()
    text=re.sub(r'\n+', '\n',text).strip()
    text=re.sub(' +', ' ',text).strip()
    text=' '.join(re.findall(r"[\w']+|[-\t'”–*?;,…’!—.‘ :)\n“(]",text))
    text=re.sub('\n','[LB]',text)
    text=text.split()
    vocab_list=list(sorted(set(text),key=str.lower))
    low=[]
    for i in range(1,len(vocab_list)):
        if vocab_list[i].lower()==vocab_list[i-1].lower():
            low.append(vocab_list[i])

    for idx in range(len(text)):
        if text[idx] in low:
            text[idx]=text[idx].lower()

    return ' '.join(text)

#write and save it in clean folder
Data=cleaner(Data_Path)
if not os.path.exists:
    os.makedirs(Write_Path[:14])
Clean_File=open(Write_Path,"w")
Clean_File.writelines(Data)
Clean_File.close()