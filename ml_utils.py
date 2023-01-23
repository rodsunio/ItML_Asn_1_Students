import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns

    Methods
    -------
    setCat(catList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        catlist : list
            The list of column names that are categorical

    setNum(numList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        numlist : list
            The list of column names that are numerical
    
    missing_values(self)
        gets the sum of the null values

    detect_outliers(self, column)
        detects outliers using Z-score method

    remove_outliers(self, column, lower_bound, upper_bound)
        removes outliers

        Paramaters
        ----------
        column: the name of the column on which you want to filter out the outliers
        lower_bound:  lower bound to filter the outliers
        upper_bound:  upper bound to filter the outliers
    
    countPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned.
    
    histPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned. 

    pairPlots(self)
        generates pairplots and a regression line with the hue function in the pairplot to split the data by the target value

    corrCoefficient(self)
        generates a heatmap of correlation
        the pandas getdummies method is used in this method to convert categorical variables into dummy/indicator variables


    fullEDA()
        Displays the full EDA process. 
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []

    def info(self):
        return self.data.info()

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList

    def missing_values(self):
        return self.data.isnull().sum()
    
    def detect_outliers(self, column):
        mean = self.data[column].mean()
        std = self.data[column].std()
        outliers = []
        for i in self.data[column]:
            z_score = (i - mean) / std
            if np.abs(z_score) > 3:
                outliers.append(i)
        return outliers

    def remove_outliers(self, column, lower_bound, upper_bound):
        self.data = self.data[(self.data[column] > lower_bound) & (self.data[column] < upper_bound)]
        return self.data

    def countPlots(self, splitTarg=False, show=True):
        n = len(self.cat)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.cat:
            if splitTarg == False:
                sns.countplot(data=self.data, x=col, ax=ax[r][c])
            if splitTarg == True:
                sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure

    def histPlots(self, kde=True, splitTarg=False, show=True):
        n = len(self.num)
        cols = 2
        figure, ax = plt.subplots(math.ceil(n/cols), cols)
        r = 0
        c = 0
        for col in self.num:
            #print("r:",r,"c:",c)
            if splitTarg == False:
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
            if splitTarg == True:
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
            c += 1
            if c == cols:
                r += 1
                c = 0
        if show == True:
            figure.show()
        return figure
    
    def pairPlots(self):
        sns.pairplot(self.data, hue = self.target, kind ='reg')
        plt.show()
    
    def corrCoefficient(self):
        plt.figure(figsize=(14,14))
        temp = pd.get_dummies(self.data, drop_first=True)
        corr = temp.corr()
        sns.heatmap(corr, annot=True, cmap ='coolwarm', fmt='.2f',
            annot_kws={'size': 10}, cbar=False)

    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3, out4, out5])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        tab.set_title(3, 'Pair Plots')
        tab.set_title(4, 'Correlation Coefficient')
        display(tab)

        with out1:
            self.info()

        with out2:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)

        with out4:
            fig4 = self.pairPlots()
            plt.show(fig4)
        
        with out5:
            fig5 = self.corrCoefficient()
            plt.show(fig5)
