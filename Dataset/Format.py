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
  return text

#write and save it in clean folder
Data=cleaner(Data_Path)
if not os.path.exists:
	os.makedirs(Write_Path[:14])
Clean_File=open(Write_Path,"w")
Clean_File.writelines(Data)
Clean_File.close()