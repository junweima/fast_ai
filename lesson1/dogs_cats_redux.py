
# coding: utf-8

# # Dogs vs Cat Redux

# In this tutorial, you will learn how generate and submit predictions to a Kaggle competiton
# 
# [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
#     
#     

# To start you will need to download and unzip the competition data from Kaggle and ensure your directory structure looks like this
# ```
# utils/
#     vgg16.py
#     utils.py
# lesson1/
#     redux.ipynb
#     data/
#         redux/
#             train/
#                 cat.437.jpg
#                 dog.9924.jpg
#                 cat.1029.jpg
#                 dog.4374.jpg
#             test/
#                 231.jpg
#                 325.jpg
#                 1235.jpg
#                 9923.jpg
# ```
# 
# You can download the data files from the competition page [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) or you can download them from the command line using the [Kaggle CLI](https://github.com/floydwch/kaggle-cli).
# 
# You should launch your notebook inside the lesson1 directory
# ```
# cd lesson1
# jupyter notebook
# ```

# In[1]:


#Verify we are in the lesson1 directory
#get_ipython().magic('pwd')


# In[2]:


#Create references to important directories we will use over and over
import os, sys
current_dir = os.getcwd()
LESSON_HOME_DIR = current_dir
DATA_HOME_DIR = current_dir+'/data/redux'


# In[3]:


#Allow relative imports to directories above lesson1/
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

#import modules
from utils import *
from vgg16 import Vgg16

#Instantiate plotting tool
#In Jupyter notebooks, you will need to run this command before doing any plotting
get_ipython().magic('matplotlib inline')


# In[4]:


#jm: reset everthing before starting
get_ipython().magic('cd $DATA_HOME_DIR')
get_ipython().magic('rm -rf valid/ results/ sample/')
get_ipython().magic('mv $DATA_HOME_DIR/train/cats/*.jpg $DATA_HOME_DIR/train')
get_ipython().magic('mv $DATA_HOME_DIR/train/dogs/*.jpg $DATA_HOME_DIR/train')
get_ipython().magic('rm -rf cats/ dogs/')


# ## Action Plan
# 1. Create Validation and Sample sets
# 2. Rearrange image files into their respective directories 
# 3. Finetune and Train model
# 4. Generate predictions
# 5. Validate predictions
# 6. Submit predictions to Kaggle

# ## Create validation set and sample

# In[5]:


#Create directories
get_ipython().magic('cd $DATA_HOME_DIR')
get_ipython().magic('mkdir valid')
get_ipython().magic('mkdir results')
get_ipython().magic('mkdir -p sample/train')
get_ipython().magic('mkdir -p sample/test')
get_ipython().magic('mkdir -p sample/valid')
get_ipython().magic('mkdir -p sample/results')
get_ipython().magic('mkdir -p test/unknown')


# In[6]:


get_ipython().magic('cd $DATA_HOME_DIR/train')


# In[7]:


g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(2000): os.rename(shuf[i], DATA_HOME_DIR+'/valid/' + shuf[i])


# In[8]:


from shutil import copyfile


# In[9]:


g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(200): copyfile(shuf[i], DATA_HOME_DIR+'/sample/train/' + shuf[i])


# In[10]:


get_ipython().magic('cd $DATA_HOME_DIR/valid')


# In[11]:


g = glob('*.jpg')
shuf = np.random.permutation(g)
for i in range(50): copyfile(shuf[i], DATA_HOME_DIR+'/sample/valid/' + shuf[i])


# ## Rearrange image files into their respective directories

# In[12]:


#Divide cat/dog images into separate directories

get_ipython().magic('cd $DATA_HOME_DIR/sample/train')
get_ipython().magic('mkdir cats')
get_ipython().magic('mkdir dogs')
get_ipython().magic('mv cat.*.jpg cats/')
get_ipython().magic('mv dog.*.jpg dogs/')

get_ipython().magic('cd $DATA_HOME_DIR/sample/valid')
get_ipython().magic('mkdir cats')
get_ipython().magic('mkdir dogs')
get_ipython().magic('mv cat.*.jpg cats/')
get_ipython().magic('mv dog.*.jpg dogs/')

get_ipython().magic('cd $DATA_HOME_DIR/valid')
get_ipython().magic('mkdir cats')
get_ipython().magic('mkdir dogs')
get_ipython().magic('mv cat.*.jpg cats/')
get_ipython().magic('mv dog.*.jpg dogs/')

get_ipython().magic('cd $DATA_HOME_DIR/train')
get_ipython().magic('mkdir cats')
get_ipython().magic('mkdir dogs')
get_ipython().magic('mv cat.*.jpg cats/')
get_ipython().magic('mv dog.*.jpg dogs/')


# In[13]:


# Create single 'unknown' class for test set
get_ipython().magic('cd $DATA_HOME_DIR/test')
get_ipython().magic('mv *.jpg unknown/')


# ## Finetuning and Training

# In[14]:


get_ipython().magic('cd $DATA_HOME_DIR')

#Set path to sample/ path if desired
path = DATA_HOME_DIR + '/' #'/sample/'
test_path = DATA_HOME_DIR + '/test/' #We use all the test data
results_path=DATA_HOME_DIR + '/results/'
train_path=path + '/train/'
valid_path=path + '/valid/'


# In[15]:


#import Vgg16 helper class
vgg = Vgg16()


# In[16]:


#Set constants. You can experiment with no_of_epochs to improve the model
batch_size=64
no_of_epochs=3


# In[17]:


#Finetune the model
batches = vgg.get_batches(train_path, batch_size=batch_size)
val_batches = vgg.get_batches(valid_path, batch_size=batch_size*2)
vgg.finetune(batches)

#Not sure if we set this for all fits
vgg.model.optimizer.lr = 0.01


# In[ ]:


#Notice we are passing in the validation dataset to the fit() method
#For each epoch we test our model against the validation set
latest_weights_filename = None
for epoch in range(no_of_epochs):
    print ("Running epoch: %d" % epoch)
    vgg.fit(batches, val_batches, nb_epoch=1)
    latest_weights_filename = 'ft%d.h5' % epoch
    vgg.model.save_weights(results_path+latest_weights_filename)
print ("Completed %s fit operations" % no_of_epochs)



