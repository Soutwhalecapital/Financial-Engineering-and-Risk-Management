
# coding: utf-8

# # Financial Engineering and Risk Management Part III
# ---
# ## Interest Rate Instruments Notebook 1
# ---
# This notebook serves as a guide for answering the programming questions of the quiz "Interest Rate Instruments Assignment Part III". The basic structure of the code has been provided and you will only be required to code some of the missing parts.
# 
# The main objective of this assignment is to do a basic data analysis on the Libor rates data (similar to the one done in class for swap rates). After completing this notebook, you will be able to:
# + Load a csv (Excel-like) data file into Python.
# + Quickly find any Libor rate for a given date.
# + Retrieve the Libor rates for a given time window.
# + Visualize Libor rates data via scatter plots.
# + Compute the historical correlation between two Libor rates for a given time window.
# 
# Most of the code is similar to the one used by Prof. Hirsa in the videos. We invite you to carefully study the lectures before going through the notebook.
# 
# Once that you are confident that the notebook is running correctly, please input your answers manually on the quiz "Interest Rate Instruments Assignment Part III".
# 
# If you wish to run this notebook (as it is) on your local computer you will need to have installed Python 3.6, Jupyter notebooks, and the following Python packages:
# 
# * numpy
# * pandas
# * matplotlib
# * sklearn
# 
# You will also need to download the csv file "swapLiborData.csv" and place it in the same folder as this Jupyter notebook.

# ### Import Python modules

# In[ ]:


import numpy as np # for fast vector computations
import pandas as pd # for easy data analysis
import matplotlib.pyplot as plt # for plotting
from sklearn import linear_model # for linear regression


# ### Reading data from a csv file and doing some preprocessing

# We load the swap and libor rates data into a Pandas dataframe. The data is in the csv file 'swapLiborData.csv' and it should be downloaded and placed in the same folder as this Jupyter notebook if you want to run it in your local environment.

# In[ ]:


# load data using pandas
df = pd.read_csv('swapLiborData.csv')


# We print the first rows of a dataframe.

# In[ ]:


df.head()


# The 'Date' variable in the original file is in numeric format. We transform it into a more readable year-month-day format.

# In[ ]:


df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Date'],'D')
df.head()


# Much better!

# ### Finding some Libor rates and plotting curves
# 
# #### Questions 1, 2, and 3 of the quiz "Interest Rate Instruments Assignment Part III"
# 

# Write a function that given the dataframe and a date such as '2014-01-08' returns a vector of length 5 
# with the libor rates for that date (1, 2, 3, 6, and 12 months). You can assume that the date given can always be found in the data.

# In[ ]:


def libor_rates_date(df, date):
    ''' Retrieve the Libor rates (all terms) for a given date. '''
    
    libor_rates = np.zeros(5)
    
    ###############################################################
    ###############################################################
    # your code starts here
    ###############################################################

    ###############################################################
    # your code ends here
    ###############################################################
    ###############################################################
    
    return libor_rates


# Let's plot the libor rates for these 5 dates:

# In[ ]:


dates = ['2014-03-13', '2014-12-29', '2016-10-07', '2017-12-13', '2018-07-20']


# In[ ]:


plt.figure(figsize=(8,6)) # you can change the size to fit better your screen

for d in dates:
    plt.plot([1, 2, 3, 6 ,12], libor_rates_date(df, d)) # plot rates

# labels, title and legends
plt.xlabel('LIBOR term')
plt.ylabel('LIBOR rate')
plt.title('LIBOR Curve on various dates')
plt.legend(dates)

plt.show()


# Now, write another function that returns the Libor rate for a specific date and term (in months). 
# The term can be any of the following integers: 1, 2, 3, 6, 12. 
# Hint: you can use the previous function 'libor_rates_date' as part of your code.

# In[ ]:


def libor_rate_date_term(df, date, term):
    ''' Retrieve the Libor rate for a given date and term. '''
    
    libor = 0.
    
    ###############################################################
    ###############################################################
    # your code starts here
    ###############################################################

    ###############################################################
    # your code ends here
    ###############################################################
    ###############################################################
   
    return libor


# Use the previous function to find the following swap rates. Input them manually on the quiz page (rounded to **4** decimals). Of course! You could manually inspect the table to find the correct rate, but that would not be any fun!

# #### Question 1
# Input manually on the quiz page.

# In[ ]:


# question 1
date = '2015-03-31'
term = 2
libor_rate = libor_rate_date_term(df, date, term)
print(np.round(libor_rate,4))


# #### Question 2
# Input manually on the quiz page.

# In[ ]:


# question 2
date = '2017-12-12'
term = 6
libor_rate = libor_rate_date_term(df, date, term)
print(np.round(libor_rate,4))


# #### Question 3
# Input manually on the quiz page.

# In[ ]:


# question 3
date = '2018-05-25'
term = 12
libor_rate = libor_rate_date_term(df, date, term)
print(np.round(libor_rate,4))


# ### Computing Libor rates correlations
# 
# #### Questions 4, 5, and 6 of the quiz "Interest Rate Instruments Assignment Part III"

# We are now interested in computing correlations between different Libor rates over certain time windows. For this analysis, write a function that given two dates d1 <= d2, returns a dataframe with all the libor rates in that time interval.

# In[ ]:


def libor_rates_time_window(df, d1, d2):
    ''' Retrieve the Libor rates (all terms) for the date window d1 to d2. '''
    
    sub_df = pd.DataFrame()
    
    ###############################################################
    ###############################################################
    # your code starts here
    ###############################################################

    ###############################################################
    # your code ends here
    ###############################################################
    ###############################################################
    
    return sub_df


# Let's use the previous function to do some scatter plots.

# In[ ]:


def scatter_plot_window(df, d1, d2):
    ''' Plots scatter plots for a time window. '''
    
    df_sub = libor_rates_time_window(df, d1, d2)
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(2,2,1)
    plt.title('Time window: ' + d1 + ' to ' + d2)
    plt.plot(df_sub.US0001M, df_sub.US0002M, '.')
    plt.xlabel('1M LIBOR rate')
    plt.ylabel('2M LIBOR rate')
    
    plt.subplot(2,2,2)
    plt.plot(df_sub.US0006M, df_sub.US0012M, '.')
    plt.xlabel('6M LIBOR rate')
    plt.ylabel('12M LIBOR rate')
    
    plt.subplot(2,2,3)
    plt.plot(df_sub.US0001M, df_sub.US0012M, '.')
    plt.xlabel('1M LIBOR rate')
    plt.ylabel('12M LIBOR rate')
    
    plt.subplot(2,2,4)
    plt.plot(df_sub.US0003M, df_sub.US0006M, '.')
    plt.xlabel('3M LIBOR rate')
    plt.ylabel('6M LIBOR rate')
    
    plt.show()
    


# For example, let's see the scatter plots for 2017.

# In[ ]:


scatter_plot_window(df, '2017-01-01', '2017-12-31')


# And now for 2018.

# In[ ]:


scatter_plot_window(df, '2018-01-01', '2018-10-11')


# Finally, write a function that given a time window [d1, d2] and two Libor terms (in months) t1 and t2 returns the correlation of the corresponding Libor rates during that window. t1 and t2 can only take the values: 1,2,3,6,12. **Hint**: use the previous function 'libor_rates_time_window' and the Pandas function **pd.df.corr()** to compute the correlation.

# In[ ]:


def corr_window(df, d1, d2, term1, term2):
    
    corr = 0.0
    
    ###############################################################
    ###############################################################
    # your code starts here
    ###############################################################

    ###############################################################
    # your code ends here
    ###############################################################
    ############################################################### 
    
    return corr


# Use the previous function to compute the following correlations. Input them manually on the quiz page (rounded to **3** decimals).

# #### Question 4
# Input manually on the quiz page.

# In[ ]:


# question 4
date1 = '2014-01-01'
date2 = '2015-12-31'
libor_term1 = 1
libor_term2 = 3
corr = corr_window(df, date1, date2, libor_term1, libor_term2)
print(np.round(corr, 3))


# #### Question 5
# Input manually on the quiz page.

# In[ ]:


# question 5
date1 = '2016-01-01'
date2 = '2017-12-31'
libor_term1 = 1
libor_term2 = 12
corr = corr_window(df, date1, date2, libor_term1, libor_term2)
print(np.round(corr, 3))


# #### Question 6
# Input manually on the quiz page.

# In[ ]:


# question 6
date1 = '2018-01-01'
date2 = '2018-10-11'
libor_term1 = 2
libor_term2 = 6
corr = corr_window(df, date1, date2, libor_term1, libor_term2)
print(np.round(corr, 3))

