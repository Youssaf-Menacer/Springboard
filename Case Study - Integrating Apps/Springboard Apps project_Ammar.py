#!/usr/bin/env python
# coding: utf-8

# # Springboard Apps project - Tier 1 - Complete
# 
# Welcome to the Apps project! To give you a taste of your future career, we're going to walk through exactly the kind of notebook that you'd write as a data scientist. In the process, we'll be sure to signpost the general framework for our investigation - the Data Science Pipeline - as well as give reasons for why we're doing what we're doing. We're also going to apply some of the skills and knowledge you've built up in the previous unit when reading Professor Spiegelhalter's *The Art of Statistics* (hereinafter *AoS*). 
# 
# So let's get cracking!
# 
# **Brief**
# 
# Did Apple Store apps receive better reviews than Google Play apps?
# 
# ## Stages of the project
# 
# 1. Sourcing and loading 
#     * Load the two datasets
#     * Pick the columns that we are going to work with 
#     * Subsetting the data on this basis 
#  
#  
# 2. Cleaning, transforming and visualizing
#     * Check the data types and fix them
#     * Add a `platform` column to both the `Apple` and the `Google` dataframes
#     * Changing the column names to prepare for a join 
#     * Join the two data sets
#     * Eliminate the `NaN` values
#     * Filter only those apps that have been reviewed at least once
#     * Summarize the data visually and analytically (by the column `platform`)  
#   
#   
# 3. Modelling 
#     * Hypothesis formulation
#     * Getting the distribution of the data
#     * Permutation test 
# 
# 
# 4. Evaluating and concluding 
#     * What is our conclusion?
#     * What is our decision?
#     * Other models we could have used. 
#     

# ## Importing the libraries
# 
# In this case we are going to import pandas, numpy, scipy, random and matplotlib.pyplot

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# scipi is a library for statistical tests and visualizations 
from scipy import stats
# random enables us to generate random numbers
import random


# ## Stage 1 -  Sourcing and loading data

# ### 1a. Source and load the data
# Let's download the data from Kaggle. Kaggle is a fantastic resource: a kind of social medium for data scientists, it boasts projects, datasets and news on the freshest libraries and technologies all in one place. The data from the Apple Store can be found [here](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps) and the data from Google Store can be found [here](https://www.kaggle.com/lava18/google-play-store-apps).
# Download the datasets and save them in your working directory.

# In[4]:


# Now that the files are saved, we want to load them into Python using read_csv and pandas.

# Create a variable called google, and store in it the path of the csv file that contains your google dataset. 
# If your dataset is in the same folder as this notebook, the path will simply be the name of the file. 
google ='googleplaystore.csv'

# Read the csv file into a data frame called Google using the read_csv() pandas method.
Google = pd.read_csv(google)

# Using the head() pandas method, observe the first three entries.
Google.head()


# In[5]:


# Create a variable called apple, and store in it the path of the csv file that contains your apple dataset. 
apple ='AppleStore.csv'

# Read the csv file into a pandas DataFrame object called Apple.
Apple = pd.read_csv(apple)

# Observe the first three entries like you did with your other data. 
Apple.head()


# ### 1b. Pick the columns we'll work with
# 
# From the documentation of these datasets, we can infer that the most appropriate columns to answer the brief are:
# 
# 1. Google:
#     * `Category` # Do we need this?
#     * `Rating`
#     * `Reviews`
#     * `Price` (maybe)
# 2. Apple:    
#     * `prime_genre` # Do we need this?
#     * `user_rating` 
#     * `rating_count_tot`
#     * `price` (maybe)

# ### 1c. Subsetting accordingly
# 
# Let's select only those columns that we want to work with from both datasets. We'll overwrite the subsets in the original variables.

# In[6]:


# Subset our DataFrame object Google by selecting just the variables ['Category', 'Rating', 'Reviews', 'Price']
Google= Google[['Category', 'Rating', 'Reviews', 'Price']]

# Check the first three entries
Google.head(3)


# In[7]:


# Do the same with our Apple object, selecting just the variables ['prime_genre', 'user_rating', 'rating_count_tot', 'price']
Apple =Apple[['prime_genre', 'user_rating', 'rating_count_tot', 'price']]

# Let's check the first three entries
Apple.head(3)


# ## Stage 2 -  Cleaning, transforming and visualizing

# ### 2a. Check the data types for both Apple and Google, and fix them
# 
# Types are crucial for data science in Python. Let's determine whether the variables we selected in the previous section belong to the types they should do, or whether there are any errors here. 

# In[9]:


# Using the dtypes feature of pandas DataFrame objects, check out the data types within our Apple dataframe.
# Are they what you expect?
Apple.dtypes


# We observe that the datatypes are what we expect.

# This is looking healthy. But what about our Google data frame?

# In[10]:


# Using the same dtypes feature, check out the data types of our Google dataframe. 
Google.dtypes


# We obeserve that Google['Price'].dtypes is weird. It should be numerical. Also, Googe['Reviews'] should be numerical too.

# Weird. The data type for the column 'Price' is 'object', not a numeric data type like a float or an integer. Let's investigate the unique values of this column. 

# In[11]:


# Use the unique() pandas method on the Price column to check its unique values. 
Google['Price'].unique()


# Aha! Fascinating. There are actually two issues here. 
# 
# - Firstly, there's a price called `Everyone`. That is a massive mistake! 
# - Secondly, there are dollar symbols everywhere! 
# 
# 
# Let's address the first issue first. Let's check the datapoints that have the price value `Everyone`

# In[12]:


# Let's check which data points have the value 'Everyone' for the 'Price' column by subsetting our Google dataframe.

# Subset the Google dataframe on the price column. 
# To be sure: you want to pick out just those rows whose value for the 'Price' column is just 'Everyone'. 
Google[Google['Price']=='Everyone']


# Thankfully, it's just one row. We've gotta get rid of it. 

# In[13]:


# Let's eliminate that row. 

# Subset our Google dataframe to pick out just those rows whose value for the 'Price' column is NOT 'Everyone'. 
# Reassign that subset to the Google variable. 
# You can do this in two lines or one. Your choice! 
Google = Google[Google['Price'] !='Everyone']

# Check again the unique values of Google
Google['Price'].unique()


# Our second problem remains: I'm seeing dollar symbols when I close my eyes! (And not in a good way). 
# 
# This is a problem because Python actually considers these values strings. So we can't do mathematical and statistical operations on them until we've made them into numbers. 

# In[14]:


# Let's create a variable called nosymb.
# This variable will take the Price column of Google and apply the str.replace() method. 
# Remember: we want to find '$' and replace it with nothing, so we'll have to write approrpiate arguments to the method to achieve this. 
nosymb = Google['Price'].str.replace('$','')

# Now we need to do two things:
# i. Make the values in the nosymb variable numeric using the to_numeric() pandas method.
# ii. Assign this new set of numeric, dollar-sign-less values to Google['Price']. 
# You can do this in one line if you wish.
Google['Price'] = pd.to_numeric(nosymb)


# Now let's check the data types for our Google dataframe again, to verify that the 'Price' column really is numeric now.

# In[16]:


# Use the function dtypes. 
Google.dtypes


# Notice that the column `Reviews` is still an object column. We actually need this column to be a numeric column, too. 

# In[20]:


# Convert the 'Reviews' column to a numeric data type. 
# Use the method pd.to_numeric(), and save the result in the same column.
Google['Reviews'] = pd.to_numeric(Google['Reviews'])


# In[21]:


# Let's check the data types of Google again
Google.dtypes


# ### 2b. Add a `platform` column to both the `Apple` and the `Google` dataframes
# Let's add a new column to both dataframe objects called `platform`: all of its values in the Google dataframe will be just 'google', and all of its values for the Apple dataframe will be just 'apple'. 
# 
# The reason we're making this column is so that we can ultimately join our Apple and Google data together, and actually test out some hypotheses to solve the problem in our brief. 

# In[22]:


# Create a column called 'platform' in both the Apple and Google dataframes. 
# Add the value 'apple' and the value 'google' as appropriate. 
Apple['platform'] = 'apple'
Google['platform'] = 'google'


# In[23]:


Apple.head()


# In[24]:


Google.head()


# ### 2c. Changing the column names to prepare for our join of the two datasets 
# Since the easiest way to join two datasets is if they have both:
# - the same number of columns
# - the same column names
# we need to rename the columns of `Apple` so that they're the same as the ones of `Google`, or vice versa.
# 
# In this case, we're going to change the `Apple` columns names to the names of the `Google` columns. 
# 
# This is an important step to unify the two datasets!

# In[28]:


# Create a variable called old_names where you'll store the column names of the Apple dataframe. 
# Use the feature .columns.
old_names = Apple.columns

# Create a variable called new_names where you'll store the column names of the Google dataframe. 
new_names = Google.columns

# Use the rename() DataFrame method to change the columns names. 
# In the columns parameter of the rename() method, use this construction: dict(zip(old_names,new_names)).
Apple = Apple.rename(columns =dict(zip(old_names,new_names)))


# In[31]:


# Checking Apple New Columns== Google columns?
Apple.columns==Google.columns


# ### 2d. Join the two datasets 
# Let's combine the two datasets into a single data frame called `df`.

# In[35]:


# Let's use the append() method to append Apple to Google. 
# Make Apple the first parameter of append(), and make the second parameter just: ignore_index = True.
df = Google.append(Apple, ignore_index= True)

# Using the sample() method with the number 12 passed to it, check 12 random points of your dataset.
df.sample(12)


# ### 2e. Eliminate the NaN values
# 
# As you can see there are some `NaN` values. We want to eliminate all these `NaN` values from the table.

# In[36]:


# Lets check first the dimesions of df before droping `NaN` values. Use the .shape feature. 
print(df.shape)

# Use the dropna() method to eliminate all the NaN values, and overwrite the same dataframe with the result. 
# Note: dropna() by default removes all rows containing at least one NaN. 
df =  df.dropna()

# Check the new dimesions of our dataframe. 
print(df.shape)


# In[41]:


np.sum(df[df=='Nan'])


# Perfect! We do not have 'Nan' in our dataframe, df.

# ### 2f. Filter the data so that we only see whose apps that have been reviewed at least once
# 
# Apps that haven't been reviewed yet can't help us solve our brief. 
# 
# So let's check to see if any apps have no reviews at all. 

# In[42]:


# Subset your df to pick out just those rows whose value for 'Reviews' is equal to 0. 
# Do a count() on the result. 
df[df['Reviews'] == 0].count()


# 929 apps do not have reviews, we need to eliminate these points!

# In[43]:


# Eliminate the points that have 0 reviews.
# An elegant way to do this is to assign df the result of picking out just those rows in df whose value for 'Reviews' is NOT 0.
df = df[df['Reviews'] != 0]


# In[44]:


# Checking 
df[df['Reviews'] == 0].count()


# ### 2g. Summarize the data visually and analytically (by the column `platform`)

# What we need to solve our brief is a summary of the `Rating` column, but separated by the different platforms.

# In[48]:


# To summarize analytically, let's use the groupby() method on our df.
# For its parameters, let's assign its 'by' parameter 'platform', and then make sure we're seeing 'Rating' too. 
# Finally, call describe() on the result. We can do this in one line, but this isn't necessary. 
df.groupby(by='platform')['Rating'].describe()


# Interesting! Our means of 4.049697 and 4.191757 don't **seem** all that different! Perhaps we've solved our brief already: there's no significant difference between Google Play app reviews and Apple Store app reviews. We have an ***observed difference*** here: which is simply (4.191757 - 4.049697) = 0.14206. This is just the actual difference that we observed between the mean rating for apps from Google Play, and the mean rating for apps from the Apple Store. Let's look at how we're going to use this observed difference to solve our problem using a statistical test. 
# 
# **Outline of our method:**
# 1. We'll assume that platform (i.e, whether the app was Google or Apple) really doesn’t impact on ratings. 
# 
# 
# 2. Given this assumption, we should actually be able to get a difference in mean rating for Apple apps and mean rating for Google apps that's pretty similar to the one we actually got (0.14206) just by: 
# a. shuffling the ratings column, 
# b. keeping the platform column the same,
# c. calculating the difference between the mean rating for Apple and the mean rating for Google. 
# 
# 
# 3. We can make the shuffle more useful by doing it many times, each time calculating the mean rating for Apple apps and the mean rating for Google apps, and the difference between these means. 
# 
# 
# 4. We can then take the mean of all these differences, and this will be called our permutation difference. This permutation difference will be great indicator of what the difference would be if our initial assumption were true and platform really doesn’t impact on ratings. 
# 
# 
# 5. Now we do a comparison. If the observed difference looks just like the permutation difference, then we stick with the claim that actually, platform doesn’t impact on ratings. If instead, however, the permutation difference differs significantly from the observed difference, we'll conclude: something's going on; the platform does in fact impact on ratings. 
# 
# 
# 6. As for what the definition of *significantly* is, we'll get to that. But there’s a brief summary of what we're going to do. Exciting!
# 
# If you want to look more deeply at the statistics behind this project, check out [this resource](https://www.springboard.com/archeio/download/4ea4d453b0b84014bcef287c50f47f00/).

# Let's also get a **visual summary** of the `Rating` column, separated by the different platforms. 
# 
# A good tool to use here is the boxplot!

# In[49]:


# Call the boxplot() method on our df.
# Set the parameters: by = 'platform' and column = ['Rating'].
df.boxplot(by='platform', column =['Rating'], grid=False, rot=45, fontsize=15)


# Here we see the same information as in the analytical summary, but with a boxplot. Can you see how the boxplot is working here? If you need to revise your boxplots, check out this this [link](https://www.kaggle.com/ramamet4/app-store-apple-data-set-10k-apps). 

# ## Stage 3 - Modelling

# ### 3a. Hypothesis formulation
# 
# Our **Null hypothesis** is just:
# 
# **H<sub>null</sub>**: the observed difference in the mean rating of Apple Store and Google Play apps is due to chance (and thus not due to the platform).
# 
# The more interesting hypothesis is called the **Alternate hypothesis**:
# 
# **H<sub>alternative</sub>**: the observed difference in the average ratings of apple and google users is not due to chance (and is actually due to platform)
# 
# We're also going to pick a **significance level** of 0.05. 

# ### 3b. Getting the distribution of the data
# Now that the hypotheses and significance level are defined, we can select a statistical test to determine which hypothesis to accept. 
# 
# There are many different statistical tests, all with different assumptions. You'll generate an excellent judgement about when to use which statistical tests over Data Science Career Track course. But in general, one of the most important things to determine is the **distribution of the data**.   

# In[50]:


# Create a subset of the column 'Rating' by the different platforms.
# Hint: this will need to have the form: apple = df[df['platform'] == 'apple']['Rating']
# Call the subsets 'apple' and 'google' 
apple = df[df['platform'] == 'apple']['Rating']
google =df[df['platform'] == 'google']['Rating']


# In[57]:


# Checking
google.head(),apple.head()


# In[59]:


# Using the stats.normaltest() method, get an indication of whether the apple data are normally distributed
# Save the result in a variable called apple_normal, and print it out
# Since the null hypothesis of the normaltest() is that the data is normally distributed, the lower the p-value in the result of this test, the more likely the data are to be normally distributed.
apple_normal = stats.normaltest(apple)
print(apple_normal)


# In[58]:


# Do the same with the google data. 
# Save the result in a variable called google_normal
google_normal = stats.normaltest(google)
print(google_normal)


# Since the null hypothesis of the normaltest() is that the data are normally distributed, the lower the p-value in the result of this test, the more likely the data are to be non-normal. 
# 
# Since the p-values is 0 for both tests, regardless of what we pick for the significance level, our conclusion is that the data are not normally distributed. 
# 
# We can actually also check out the distribution of the data visually with a histogram. A normal distribution has the following visual characteristics:
#     - symmetric
#     - unimodal (one hump)
# As well as a roughly identical mean, median and mode. 

# In[60]:


# Create a histogram of the apple reviews distribution
# You'll use the plt.hist() method here, and pass your apple data to it
histoApple = plt.hist(apple)


# In[62]:


# Create a histogram of the google data
histoGoogle =plt.hist(google)


# ### 3c. Permutation test
# Since the data aren't normally distributed, we're using a *non-parametric* test here. This is simply a label for statistical tests used when the data aren't normally distributed. These tests are extraordinarily powerful due to how few assumptions we need to make.  
# 
# Check out more about permutations [here.](http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/)

# In[63]:


# Create a column called `Permutation1`, and assign to it the result of permuting (shuffling) the Rating column
# This assignment will use our numpy object's random.permutation() method, and will look like this:
# df['Permutation1'] = np.random.permutation(df['Rating'])
df['Permutation1'] = np.random.permutation(df['Rating'])

# Call the describe() method on our permutation grouped by 'platform'. 
# We'll use this structure: df.groupby(by='platform')['Permutation1'].describe()
df.groupby(by='platform')['Permutation1'].describe()


# In[64]:


# Lets compare with the previous analytical summary: use df.groupby(by='platform')['Rating'].describe()
df.groupby(by='platform')['Rating'].describe()


# In[66]:


# The difference in the means for Permutation1 (0.001103) now looks hugely different to our observed difference of 0.14206. 
# It's sure starting to look like our observed difference is significant, and that the Null is false; platform does impact on ratings
# But to be sure, let's create 10,000 permutations, calculate the mean ratings for Google and Apple apps and the difference between these for each one, and then take the average of all of these differences.
# Let's create a vector with the differences - that will be the distibution of the Null.

# First, make a list called difference.
difference= list()

# Now make a for loop that does the following 10,000 times:
# 1. makes a permutation of the 'Rating' as you did above
# 2. calculates the difference in the mean rating for apple and the mean rating for google. 
# Hint: the code for (2) will look like this: difference.append(np.mean(permutation[df['platform']=='apple']) - np.mean(permutation[df['platform']=='google']))
for i in range(10000):
    permutation = np.random.permutation(df['Rating'])
    difference.append(np.mean(permutation[df['platform']=='apple']) - np.mean(permutation[df['platform']=='google']))


# In[67]:


# Make a variable called 'histo', and assign to it the result of plotting a histogram of the difference list. 
# This assignment will look like: histo = plt.hist(difference)
histo = plt.hist(difference)


# In[75]:


import numpy as np

# Assuming you have DataFrames named 'apple' and 'google'

# Calculate the observed difference
#obs_difference = np.mean(apple['apple']) - np.mean(google['google'])
print(np.mean(apple['Apple']))
# # Make the difference absolute
# obs_difference = abs(obs_difference)

# # Print the result
print(obs_difference)


# ## Stage 4 -  Evaluating and concluding
# ### 4a. What is our conclusion?

# In[ ]:


'''
What do we know? 

Recall: The p-value of our observed data is just the proportion of the data given the null that's at least as extreme as that observed data.

As a result, we're going to count how many of the differences in our difference list are at least as extreme as our observed difference.

If less than or equal to 5% of them are, then we will reject the Null. 
'''
positiveExtremes = []
negativeExtremes = []
for i in range(len(difference)):
    if (difference[i] >= obs_difference):
        positiveExtremes.append(difference[i])
    elif (difference[i] <= -obs_difference):
        negativeExtremes.append(difference[i])

print(len(positiveExtremes))
print(len(negativeExtremes))


# ### 4b. What is our decision?
# So actually, zero differences are at least as extreme as our observed difference!
# 
# So the p-value of our observed data is 0. 
# 
# It doesn't matter which significance level we pick; our observed data is statistically significant, and we reject the Null.
# 
# We conclude that platform does impact on ratings. Specifically, we should advise our client to integrate **only Google Play** into their operating system interface. 

# ### 4c. Other statistical tests, and next steps
# The test we used here is the Permutation test. This was appropriate because our data were not normally distributed! 
# 
# As we've seen in Professor Spiegelhalter's book, there are actually many different statistical tests, all with different assumptions. How many of these different statistical tests can you remember? How much do you remember about what the appropriate conditions are under which to use them? 
# 
# Make a note of your answers to these questions, and discuss them with your mentor at your next call. 
# 
