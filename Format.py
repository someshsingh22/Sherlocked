#imports
import re
import os

#set path for file
Data_Path='Dataset/cano.txt'
Write_Path=Data_Path[:8]+"Clean/"+Data_Path[8:]

#clean file
def cleaner(file_path):
  text = open(file_path).read()
  text=re.sub(r'\n+', '\n',text).strip()
  text=re.sub(' +', ' ',text).strip()
  return text

#write and save it in clean folder
Data=cleaner(Data_Path)
if not os.path.exists:
	os.makedirs(Write_Path[:14])
Clean_File=open(Write_Path,"w")
Clean_File.writelines(Data)
Clean_File.close()