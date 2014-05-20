# Preliminary Report


## 1. Current work

### 1.1 Defining the problem

Information is given on a training set of passengers of the Titanic, for which the survival outcome is known. Given the training set information, the challenge is to predict each passenger’s survival outcome from a test set of passengers. The details of the challenge are given on the Kaggle site.

We will apply machine learning tools to solve the problem. In this case, we may consider approaches based on random forests. And will use Python package to plot the result. The python package used includes:
	
	NumPy
	Pandas
	SciKit-Learn
	SciPy
	StatsModels
	Patsy
	Matplotlib


### 1.2 Analyze the data
In the training dataset, there are 891 passengers. Each passenger has 12 attributes. Except the "passangerId" and "Survived" attribute, we need to consider 10 other attributes and predict the survival possibility.

#### 1.2.1 Take care of missing values:
The features *ticket* and *cabin* have many missing values and so can’t add much value to our analysis. To handle this we will drop them from the data frame to preserve the integrity of our dataset. What's more, we will remove NaN values from every remaining column. Using **drop()** and **dropna()** function in could easily achieve goal. Now we have a clean and tidy dataset that is ready for analysis, we cut the dataset from 891 to 712, and get 8 effective attributes to do prediction.

#### 1.2.2 Graphically view of data
The point of this competition is to predict if an individual will survive based on the features in the data like:

* Traveling Class (called pclass in the data)
* Sex
* Age





### 1.3 Simple trying

### 1.4 Supervised machine learning

## 2 Next step

### 2.1 SVM 

### 2.2 Random Forest

## Possible problems
