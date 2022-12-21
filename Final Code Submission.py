#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
from PIL import Image
from PIL import ImageOps
import PIL.Image   
import imagehash
import cv2
import glob
import os
from os import listdir,makedirs
from os.path import isfile,join
import tensorflow as tf
import numpy as np
import cv2 as cv
import numpy
import ipyplot
import pytesseract
from scipy.spatial.distance import hamming
#time computation
from time import perf_counter
import wget
from datetime import date, datetime, timedelta
import pandas as pd
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from PIL import ImageFont
from PIL import ImageDraw
import turtle
#file copying
import shutil
import ipyplot
import requests
import webbrowser
#pulling data from HTML and XML files
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import os


# In[2]:


#Image manipulation technique- Resizing Image 
path = r"./Pictures/Resized_Images/"
dirs= os.listdir(path)
for file in os.listdir(path):
    try:
        path_image = path+"/"+file
        image = Image.open(path_image)
        #image re-sized to 600*600 resolution
        image = image.resize((600,600))
        image.save(path_image)
    except:
        print("{} is not resized".format(image))


# In[3]:


# Image Manipulation Technique 2
#Gray Scale Technique
path = "./Pictures/News/"


# In[4]:


dest_path = "./Pictures/GrayScale Images/"


# In[5]:


#LA -8 bit pixels with black and white
for filename in os.listdir(path): 
    try:
        img = Image.open(path + filename).convert("LA")                                                                                                                                                
        img.save(dest_path + filename) 
    except:
        print ("{} is not converted".format(image))


# In[10]:


#Image Manipulation Technique 3
#Adding watermark to the image
path = './Pictures/News/'


# In[11]:


dest_path = './Pictures/Watermark/'


# In[12]:


#Logo is embedded in the images
for jpgfile in glob.iglob(os.path.join(path, "*.png")):
    shutil.copy(jpgfile, dest_path)
    logo = Image.open('Logo.png').convert('RGBA')
    photos = glob.glob('./Pictures/Watermark/*.png')
for i in photos :
    path = i.split('\\')[-1]
    img = Image.open(i)
    #pasting the logo in the image file
    img.paste(logo,(60,60),logo)
    img.save('' + str(i))


# In[13]:


# Renaming Watermarked Images
folder = './Pictures/Watermark/'
count = 1
for file_name in os.listdir(folder):
    # Old file name
    source = folder + file_name

    # New file name
    destination = folder + "watermark_" + str(count) + ".png"
    os.rename(source, destination)
    count += 1

print('New file names are')
res = os.listdir(folder)
print(res)


# In[14]:


# Perceptual hashes extracted from the image
def get_phash(path):
    hfunc = imagehash.phash
    f = []
    t = 0
    s = 0
    image_filenames = []
    #Iterated through each file in the directory
    image_filenames += [file for file in os.listdir(path)]
    #creating an empty dictionary to store the images
    images = {}
    for img in image_filenames:
        # Appending the files of the extensiosns to the dictionary
        if img.split(".")[-1] in ["jpg", "jpeg", "png"]:
            try:
                phash = hfunc(Image.open(os.path.join(path, img)).convert("RGBA"))
                images[img] = str(phash)
                s +=1
                #function returning the pHash directorty containing the filenames
            except Exception:
                f.append(img)
                continue
            t +=1
    print("phash generated for {} of {} images".format(s, t))
    return images


# In[15]:


#perf counter for computation
start = perf_counter()
hashes = get_phash(os.getcwd()+"\Pictures\images")
compute = perf_counter() - start
print(compute)


# In[16]:


# dataframe for extracted phash
df = pd.DataFrame.from_dict(hashes, orient="index", columns = ["phash"])
df.reset_index(inplace=True)
df.rename(columns={"index":"filename"}, inplace=True)
df.head(21)


# In[17]:


df.dtypes


# In[18]:


df.groupby(by=["phash"]).count()


# In[19]:


df.groupby(by=["phash"]).sum(" ")


# In[20]:


df2 = df.groupby(['phash'])['phash'].count()


# In[21]:


df2 = df.groupby(['phash']).size().reset_index(name='counts')


# In[22]:


df2.dtypes


# In[23]:


df.dtypes


# In[24]:


df3 = pd.merge(df2,df,how='left', on=['phash'])


# In[25]:


df3.head


# In[26]:


#data frame created for the pHash images
grouped = df3.groupby(by="phash").agg({"filename":"size"})
grouped.rename(columns={'filename':'count'},inplace =True)
sorted = df3.sort_values("counts", ascending= False)
sorted.head(5)


# In[27]:


len(sorted)


# In[28]:


# clustering with atleast 2 images
print("Number of clusters when minimum number of images is 2:", len(sorted[sorted["counts"] >=2]))


# In[29]:


identical_clusters = df3["counts"] >= 2
identical_clusters.reset_index(inplace=True,drop = True)
len(identical_clusters)


# In[30]:


#pHash of the manipulated images
for i, row in df3.iterrows():
    print(row["phash"])


# In[31]:


#Query pHash value
q_image = "815c75e31cc361be"
matches = {}
for i, row in df3.iterrows():
    # converting the hash hex string to binary hash value
    diff = imagehash.hex_to_hash(q_image) - imagehash.hex_to_hash(row["phash"])
    #threshold value of hamming distance is set to 10
    
    if diff <= 10:
        print("Found match: ", row["phash"])
        print("Hamming distance from query:", diff)
        matches[row["phash"]] = row["filename"]
        print("")


# In[32]:


print(matches)


# In[33]:


#Clustering technique of the pHash Value
images = [ ]
for i in matches.values():
    print("For loop 1")
    img_path = "pictures/images/" + i
    images.append(img_path)

# Load and display the images
img_arrays = []
for img in images:
    img_arrays.append(mpimg.imread(img))

plt.rcParams["figure.figsize"] = (10,8)
columns = 3
for i, image in enumerate(img_arrays):
    print("Line 14")
    plt.subplot(int(len(images) / columns + 1), columns, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)


# In[35]:


#Image extracted from the web
url = input("Enter URL: ")

headers = {
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en-US,en;q=0.8',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Referer': 'http://www.wikipedia.org/',
    'Connection': 'keep-alive',
}

r = requests.get(url=url, headers=headers)
#print(r)
soup = BeautifulSoup(r.text, 'html.parser')

i = -1
folder="Webscrape_Pic1"
os.makedirs(folder)
for img in soup.findAll('img'):
    i = i + 1

    image_path = img.get('src')
    

    if '.jpg' in image_path:
        with open(f"{folder}/{i}.jpg", 'wb') as f:
            f.write(requests.get(image_path).content)
    else:
        pass

    if '.png' in image_path:
        with open(f"{folder}/{i}.png", 'wb') as f:
            f.write(requests.get(image_path).content)
    else:
        pass

    if '.webp' in image_path:
        with open(f"{folder}/{i}.webp", 'wb') as f:
            f.write(requests.get(image_path).content)
    else:
        pass

    if '.jpeg' in image_path:
        with open(f"{folder}/{i}.jpeg", 'wb') as f:
            f.write(requests.get(image_path).content)
    else:
        pass


# In[43]:


#the manipulated images are merged with extracted images
def make_new_folder(folder_name, parent_folder):
      
   
    path = os.path.join(parent_folder, folder_name)
      
    #new folder created
    try: 
        # full access- read, write and access
        mode = 0o777
  
        # New folder created
        os.mkdir(path, mode) 
    except OSError as error: 
        print(error)
  

current_folder = os.getcwd() 
  
# Two folders to be merged
list_dir = ['images',"Image2"]
  

content_list = {}
for index, val in enumerate(list_dir):
    path = os.path.join(current_folder, val)
    content_list[ list_dir[index] ] = os.listdir(path)
  
 
# merged folder
merged_folder = "merge_folder"
  


merge_folder_path = os.path.join(current_folder, merged_folder) 
  
# if merge folder does not exists then create them
make_new_folder(merged_folder, current_folder)
  

for sub_dir in content_list:
  
   
    for contents in content_list[sub_dir]:
  
        # after looping moving through each content
        path_to_content = sub_dir + "/" + contents  
  
        # Path is set to current folder
        dir_to_move = os.path.join(current_folder, path_to_content )
  
        # path moved to merge with the folder
        shutil.move(dir_to_move, merge_folder_path)


# In[44]:


#pHash extraction of images
def get_phash(path):
    hfunc = imagehash.phash
    f = []
    t = 0
    s = 0
    image_filenames = []
    #Iterated through each file in the directory
    image_filenames += [file for file in os.listdir(path)]
    #creating an empty dictionary to store the images
    images = {}
    for img in image_filenames:
        # Appending the files of the extensiosns to the dictionary
        if img.split(".")[-1] in ["jpg", "jpeg", "png"]:
            try:
                phash = hfunc(Image.open(os.path.join(path, img)).convert("RGBA"))
                images[img] = str(phash)
                s +=1
                #function returning the pHash directorty containing the filenames
            except Exception:
                f.append(img)
                continue
            t +=1
    print("phash generated for {} of {} images".format(s, t))
    return images


# In[46]:


#time taken for computation
start = perf_counter()
hashes = get_phash(os.getcwd()+"./merge_folder")
compute = perf_counter() - start
print(compute)


# In[47]:


#dataframe with the extracted pHash
df4 = pd.DataFrame.from_dict(hashes, orient="index", columns = ["phash"])
df4.reset_index(inplace=True)
df4.rename(columns={"index":"filename"}, inplace=True)
df4.head(21)


# In[48]:


df4.dtypes


# In[49]:


df4.groupby(by=["phash"]).count()


# In[50]:


df4.groupby(by=["phash"]).count()


# In[51]:


df4.groupby(by=["phash"]).sum(" ")


# In[52]:


df5 = df4.groupby(['phash'])['phash'].count()


# In[53]:


df5 = df4.groupby(['phash']).size().reset_index(name='counts')


# In[54]:


df5.dtypes


# In[55]:


df6 = pd.merge(df5,df4,how='left', on=['phash'])


# In[56]:


df6.head


# In[57]:


#new data fram constructed with the merged-folder images
grouped = df6.groupby(by="phash").agg({"filename":"size"})
grouped.rename(columns={'filename':'count'},inplace =True)
sorted = df6.sort_values("counts", ascending= False)
sorted.head(10)


# In[58]:


len(sorted)


# In[59]:


#Output of atleast two cluters
print("Number of clusters when minimum number of images is 2:", len(sorted[sorted["counts"] >=2]))


# In[60]:


identical_clusters = df6["counts"] >= 2
identical_clusters.reset_index(inplace=True,drop = True)
len(identical_clusters)


# In[65]:


#pHash of the images
for i, row in df6.iterrows():
    print(row["phash"])


# In[66]:


#Query pHash value
q_image = "815c75e31cc361be"
matches = {}
for i, row in df3.iterrows():
    # converting the hash hex string to binary hash value
    diff = imagehash.hex_to_hash(q_image) - imagehash.hex_to_hash(row["phash"])
    #threshold value of hamming distance is set to 10
    
    if diff <= 10:
        print("Found match: ", row["phash"])
        print("Hamming distance from query:", diff)
        matches[row["phash"]] = row["filename"]
        print("")


# In[67]:


print(matches)


# In[68]:


#clustering of the images
images = [ ]
for i in matches.values():
    img_path = "pictures/images/" + i
    images.append(img_path)

# Load and display the images
img_arrays = []
for img in images:
    img_arrays.append(mpimg.imread(img))
#default seetings are brought by the rcParams element
plt.rcParams["figure.figsize"] = (10,8)
columns = 3
for i, image in enumerate(img_arrays):
    plt.subplot(int(len(images) / columns + 1), columns, i + 1)
    #It helps to plot the images in the x-axis and y-axis respectively
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)


# In[ ]:




