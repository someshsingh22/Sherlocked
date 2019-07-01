import re
import os

Data_Path='Dataset/cano.txt'
Write_Path=Data_Path[:8]+"Clean/"+Data_Path[8:]

def cleaner(file_path):
  text = open(file_path).read()
  text=re.sub(r'\n+', '\n',text).strip()
  text=re.sub(' +', ' ',text).strip()
  return text

Data=cleaner(Data_Path)
if not os.path.exists:
	os.makedirs(Write_Path[:14])

Clean_File=open(Write_Path,"w")
Clean_File.writelines(Data)
Clean_File.close()