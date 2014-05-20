
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.nonparametric import  smoothers_lowess
from pandas import Series,DataFrame
from patsy import dmatrices
import matplotlib.pyplot as plt
from pylab import *

def read_file():

    df = pd.read_csv("train.csv")
    df = df.drop(['Ticket','Cabin'], axis=1)
    df = df.dropna() # Remove NaN values
    return df

def draw_survival(df):

    plt.figure(figsize=(6,4))
    df.Survived.value_counts().plot(kind='bar', color="grey", alpha=0.35, title="Survival (1 = Survived, 0 = Died)")
    plt.ylabel("Number of Persons", fontsize=20)
    ax=plt.gca()
    ax.set_xticklabels(["Died", "Survived"], rotation=0)
    plt.savefig("survival_count.eps")

def draw_survival_on_gender(df):

    fig = plt.figure(figsize=(18,6))

    #create a plot of two subsets, male and female, of the survived variable.
    #After we do that we call value_counts() so it can be easily plotted as a bar graph.
    #'barh' is just a horizontal bar graph
    as1=fig.add_subplot(121)
    df.Survived[df.Sex == 'male'].value_counts().plot(kind='bar', color = 'black', alpha=.65, label='Male')
    df.Survived[df.Sex == 'female'].value_counts().plot(kind='bar', color='red',label='Female', alpha=.65)
    plt.ylabel("Number of Persons", fontsize=20)
    title("Survival on Gender (count)"); legend(loc='best')
    as1.set_xticklabels(["Died","Survived"], rotation=0)
    plt.savefig("survival_gender_count.eps")


def draw_survival_gender_plcass(df):
    fig=plt.figure(figsize=(18,4), dpi=1600)
    a=.65 # our alpha or opacity level.

    # building on the previous code, here we create an additional subset with in the gender subset we created for the survived variable.
    # I know, thats a lot of subsets. After we do that we call value_counts() so it it can be easily plotted as a bar graph.
    # this is repeated for each gender class pair.
    ax1=fig.add_subplot(141)
    df.Survived[df.Sex == 'male'][df.Pclass != 3].value_counts().plot(kind='bar', label='male high class', color='darkblue', alpha=a)
    ax1.set_xticklabels(["Died", "Survived"], rotation=0)
    title("Survival on Gender and Class"); legend(loc='best')

    ax2=fig.add_subplot(142, sharey=ax1)
    df.Survived[df.Sex == 'female'][df.Pclass != 3].value_counts(ascending=True).plot(kind='bar', label='female, high class', color='#FA2479', alpha=a)
    ax2.set_xticklabels(["Died","Survived"], rotation=0)
    legend(loc='best')

    ax4=fig.add_subplot(143, sharey=ax1)
    df.Survived[df.Sex == 'male'][df.Pclass == 3].value_counts().plot(kind='bar', label='male low class', alpha=a, color='lightblue')
    ax4.set_xticklabels(["Died","Survived"], rotation=0)
    legend(loc='best')

    ax3=fig.add_subplot(144, sharey=ax1)
    df.Survived[df.Sex == 'female'][df.Pclass == 3].value_counts().plot(kind='bar', label='female, low class',color='pink', alpha=a)
    ax3.set_xticklabels(["Died","Survived"], rotation=0)
    legend(loc='best')

    plt.savefig("survival_gender_class.eps")

def draw_survival_gender_age(df):
    a=.65 # our alpha or opacity level.
    fig=plt.figure(figsize=(18,4), dpi=1600)
    df.Age = df.Age.astype(float)


    ax1=fig.add_subplot(141)
    df.Survived[df.Sex == 'male'][df.Age >= 18.00].value_counts().plot(kind='bar', label='male adult', alpha=a, color='darkblue')
    ax1.set_xticklabels(["Died", "Survived"], rotation=0)
    title("Survival on Gender and Age")
    legend(loc='best')

    ax2=fig.add_subplot(142, sharey=ax1)
    df.Survived[df.Sex == 'female'][df.Age >= 18.00].value_counts(ascending=True).plot(kind='bar', label='female, adult', color='#FA2479', alpha=a)
    ax2.set_xticklabels(["Died","Survived"], rotation=0)
    legend(loc='best')

    ax3=fig.add_subplot(143, sharey=ax1)
    df.Survived[df.Sex == 'male'][df.Age < 18.00].value_counts().plot(kind='bar', label='male children', alpha=a, color='lightblue')
    ax3.set_xticklabels(["Died","Survived"], rotation=0)
    legend(loc='best')

    ax4=fig.add_subplot(144, sharey=ax1)
    df.Survived[df.Sex == 'female'][df.Age < 18.00].value_counts(ascending=True).plot(kind='bar', label='female children', alpha=a, color='pink')
    ax4.set_xticklabels(["Died","Survived"], rotation=0)
    legend(loc='best')

    plt.savefig("survival_gender_age.eps")

df = read_file()
draw_survival(df)
draw_survival_on_gender(df)
draw_survival_gender_plcass(df)
draw_survival_gender_age(df)