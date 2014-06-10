import sys
import os

import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.nonparametric import  smoothers_lowess
from statsmodels.nonparametric import kde
from pandas import Series,DataFrame
from patsy import dmatrices
import matplotlib.pyplot as plt
from pylab import *
from sklearn import datasets, svm

from statsmodels.nonparametric.kde import KDEUnivariate


# lib_path = os.popen("pwd").read()[:-1] + "/lib"
# sys.path.append(lib_path)
from lib import kaggleaux as ka




def read_file():

    df = pd.read_csv("train.csv")
    df = df.drop(['Ticket','Cabin'], axis=1)
    df = df.dropna() # Remove NaN values
    return df

def read_test():
    test_data = pd.read_csv("test.csv")
    # test_data = test_data.drop(['Ticket','Cabin'], axis=1)
    # test_data = test_data.dropna()
    return test_data

def draw_survival(df):

    plt.figure(figsize=(6,4))
    df.Survived.value_counts().plot(kind='bar', color="grey", alpha=0.35, title="Survival (1 = Survived, 0 = Died)")
    plt.ylabel("Number of Persons", fontsize=20)
    ax=plt.gca()
    ax.set_xticklabels(["Died", "Survived"], rotation=0)
    plt.savefig("tex/eps/survival_count.eps")

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
    plt.savefig("tex/eps/survival_gender_count.eps")


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

    plt.savefig("tex/eps/survival_gender_class.eps")

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

    plt.savefig("tex/eps/survival_gender_age.eps")

def draw_logit_regression(df, kind):
    w = open("logit_result.txt", "w")
    formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp  + C(Embarked)' # here the ~ sign is an = sign, and the features of our dataset
    results = {} # create a results dictionary to hold our regression results for easy analysis later
    y, x = dmatrices(formula, data=df, return_type='dataframe')
    model = sm.Logit(y, x)
    res = model.fit()
    results['Logit'] = [res, formula]
    print >> w, res.summary()

    if kind is 1:
        return results

    # Plot Predictions Vs Actual
    plt.figure(figsize=(18,4));
    plt.subplot(121, axisbg="#DBDBDB")
    # generate predictions from our fitted model
    ypred = res.predict(x)
    plt.plot(x.index, ypred, 'bo', x.index, y, 'mo', alpha=.25);
    plt.grid(color='white', linestyle='dashed')
    plt.title('Logit predictions, Blue: \nFitted/predicted values: Red');
    plt.savefig("1.eps")

    # Residuals
    plt.subplot(122, axisbg="#DBDBDB")
    plt.plot(res.resid, 'r-')
    plt.grid(color='white', linestyle='dashed')
    plt.title('Logit Residuals');
    plt.savefig("2.eps")



    fig = plt.figure(figsize=(18,9), dpi=1600)
    a = .2

    # Below are examples of more advanced plotting. 
    # It it looks strange check out the tutorial above.
    fig.add_subplot(221, axisbg="#DBDBDB")
    kde_res = KDEUnivariate(res.predict())
    kde_res.fit()
    plt.plot(kde_res.support,kde_res.density)
    plt.fill_between(kde_res.support,kde_res.density, alpha=a)
    title("Distribution of our Predictions")

    fig.add_subplot(222, axisbg="#DBDBDB")
    plt.scatter(res.predict(),x['C(Sex)[T.male]'] , alpha=a)
    plt.grid(b=True, which='major', axis='x')
    plt.xlabel("Predicted chance of survival")
    plt.ylabel("Gender Bool")
    title("The Change of Survival Probability by Gender (1 = Male)")

    fig.add_subplot(223, axisbg="#DBDBDB")
    plt.scatter(res.predict(),x['C(Pclass)[T.3]'] , alpha=a)
    plt.xlabel("Predicted chance of survival")
    plt.ylabel("Class Bool")
    plt.grid(b=True, which='major', axis='x')
    title("The Change of Survival Probability by Lower Class (1 = 3rd Class)")

    fig.add_subplot(224, axisbg="#DBDBDB")
    plt.scatter(res.predict(),x.Age , alpha=a)
    plt.grid(True, linewidth=0.15)
    title("The Change of Survival Probability by Age")
    plt.xlabel("Predicted chance of survival")
    plt.ylabel("Age")
    plt.savefig("prediction.eps")


def test_logit_regression(results):

    # lib_path = os.popen("pwd").read()[:-1] + "/lib"
    # sys.path.append(lib_path)

    test_data['Survived'] = 1.223
    print test_data

    compared_results = ka.predict(test_data, results, 'Logit') # Use your model to make prediction on our test set. 
    print compared_results
    compared_results = Series(compared_results)                 # convert our model to a series for easy output
    compared_results.to_csv("logitregres.csv")

def test_SVM(df, test_data):
    # Create an acceptable formula for our machine learning algorithms
    formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'    #set plotting parameters
    plt.figure(figsize=(8,6))

    # create a regression friendly data frame
    y, x = dmatrices(formula_ml, data=df, return_type='matrix')

    #select which features we would like to analyze
    #try chaning the selection here for diffrent output.
    #Choose : [2,3] - pretty sweet DBs [3,1] --standard DBs [7,3] -very cool DBs, [3,6] -- very long complex dbs, could take over an hour to calculate! 
    feature_1 = 2
    feature_2 = 3

    X = np.asarray(x)
    X = X[:,[feature_1, feature_2]]  


    y = np.asarray(y)
    y = y.flatten()      # needs to be 1 dimenstional so we flatten. it comes out of dmatirces with a shape. 

    n_sample = len(X)

    np.random.seed(0)
    order = np.random.permutation(n_sample)

    X = X[order]
    y = y[order].astype(np.float)

    # do a cross validation
    X_train = X[:.9 * n_sample]
    y_train = y[:.9 * n_sample]
    X_test = X[.9 * n_sample:]
    y_test = y[.9 * n_sample:]

    #create a list of the types of kerneks we will use for your analysis
    types_of_kernels = ['linear', 'rbf', 'poly']

    # specify our color map for plotting the results
    color_map = cm.RdBu_r

    # fit the model
    for fig_num, kernel in enumerate(types_of_kernels):
        clf = svm.SVC(kernel=kernel, gamma=3)
        clf.fit(X_train, y_train)

        figure(fig_num)
        #pl.clf()
        scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=color_map)

        # Circle out the test data
        scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
        
        axis('tight')
        x_min = X[:, 0].min()
        x_max = X[:, 0].max()
        y_min = X[:, 1].min()
        y_max = X[:, 1].max()

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        pcolormesh(XX, YY, Z > 0, cmap=color_map)
        contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                   levels=[-.5, 0, .5])

        title(kernel)
        plt.savefig('test_SVM_'+kernel);
        # show()
    #based on poly model
    clf = svm.SVC(kernel='poly', gamma=3).fit(X_train, y_train) 
    test_data['Survived'] = 1.23                                                         
    y,x = dmatrices(formula_ml, data=test_data, return_type='dataframe')

    res_svm = clf.predict(x.ix[:,[2,8]].dropna())                                        

    res_svm = DataFrame(res_svm,columns=['Survived'])
    res_svm.to_csv("svm_poly.csv")

    #based on rbf model
    clf = svm.SVC(kernel='rbf', gamma=3).fit(X_train, y_train) 
    test_data['Survived'] = 1.23                                                         
    y,x = dmatrices(formula_ml, data=test_data, return_type='dataframe')

    res_svm = clf.predict(x.ix[:,[2,8]].dropna())                                        

    res_svm = DataFrame(res_svm,columns=['Survived'])
    res_svm.to_csv("svm_rbf.csv")

    #based on linear model
    clf = svm.SVC(kernel='linear', gamma=3).fit(X_train, y_train) 
    test_data['Survived'] = 1.23                                                         
    y,x = dmatrices(formula_ml, data=test_data, return_type='dataframe')

    res_svm = clf.predict(x.ix[:,[2,8]].dropna())                                        

    res_svm = DataFrame(res_svm,columns=['Survived'])
    res_svm.to_csv("svm_linear.csv")

def draw_for_SVM_intro():
    # Create a new figure of size 8x6 points, using 80 dots per inch
    figure(figsize=(8,6), dpi=80)

    # Create a new subplot from a grid of 1x1
    subplot(1,1,1)

    X = np.linspace(0, 100, 256,endpoint=True)
    C = X*X*X/1000000

    # Plot cosine using blue color with a continuous line of width 1 (pixels)

    # Plot sine using green color with a continuous line of width 1 (pixels)
    plot(X, C, color="red",  linewidth=2.5, linestyle="-")

    # Set x limits
    xlim(0,100)

    # Set x ticks
    xticks(np.linspace(0,100,6,endpoint=True))

    # Set y limits
    ylim(0,1.0)

    # Set y ticks
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\frac{x^3}{10^6}$')
    yticks(np.linspace(0,1,6,endpoint=True))
    plt.title(r'$x^3$')
    plt.savefig("tex/eps/x^3.eps",dpi=72)
    # show()
    plot(X, C, color="red",  linewidth=2.5, linestyle="-")
    xticks(np.linspace(1,100,3,endpoint=True))
    yticks(np.linspace(0.000001,1,7,endpoint=True))
    # yticks(np.linspace(0,100,2,endpoint=True))
    axis((1, 100, 0.000001, 1))
    plt.xlabel(r'$x$')
    plt.ylabel(r'$log(\frac{x^3}{10^6})$')
    plt.title(r'$log(x^3)$')
    yscale('log')
    xscale('log')
    plt.savefig("tex/eps/log_x^3.eps",dpi=72)


df = read_file()
test_data = read_test()
# print test_data
# draw_survival(df)
# draw_survival_on_gender(df)
# draw_survival_gender_plcass(df)
# draw_survival_gender_age(df)
# draw_logit_regression(df,0)
# results = draw_logit_regression(df, 1)
# test_logit_regression(results)
test_SVM(df,test_data)
# draw_for_SVM_intro()



