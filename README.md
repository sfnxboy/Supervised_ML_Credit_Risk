# Supervised_ML_Credit_Risk

In this project we use Python to build and evaluate a number of machine learning models to predict credit risk. The ability to predict credit risk with machine learning algorithms helps banks and other financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud. In this project we will use machine learning to predict credit risk, this may provide a quicker and more reliable loan experience. Machine learning may also provide insight on good candidates for loans, which may lead to lower default rates. We will build and evaluate several machine learning models/algorithms to predict credit risks. Resampling and boosting are some of the techniques we will use to make the most of our models and our data.

The following setup is avoidable if you don't mind importing libraries for every program.
**Setup dependencies for Windows**

Deactivate active environment ```conda deactivate```  
Update the global conda environment ```conda update conda```  
Press the "Y" key at the prompt  
type ```conda create -n mlenv python=3.7 anaconda```  
Press the "Y" key at the prompt  
Activate the mlenv environment with ```conda activate mlenv```

**Install the imbalanced-learn package**

Now with the mlenv environment activated we can install the imbalanced-learn package with the following code:  
```conda install -c conda-forge imbalanced-learn```

Check if the package installed properly with the following code: ```conda list | findstr imbalanced-learn```

Use the following code to add the machine learning environment to Jupyter Notebook:  
```python -m ipykernel install --user --name mlenv```



## Notes

### How a Machine Learning Algorithm is Used in Data analytics

Machine learning is the use of statistical algorithms to perform tasks such as learning from data patterns and making predictions. There are many different models, a model is a mathematical representation of something that happens in the real world. Broadly speaking, machine learning can be divided into three categories: supervised, unsupervised, and deep. **Supervised learning** deals with labeled data, that is the dataset is labeled. Each row represents an instance, and each column indicated a corresponding variable. In contrast **unsupervised learning** work with dataets without labeled outcomes. An example of supervised learning might be to task a machine learning algorithm with grouping a bag of objects as it sees fit. The algorithm isn't given labels, so its on its own to find patterns. In this case, it may group the objects based on size, shape or color. In this project we will mostly talk about supervised learning. The machine learning process follows a basic pattern. First a machine learning model is presented a dataset, then the model algorithms analyze the data and attempt to identify patterns. Based on these patterns, the model makes prediction on new data.

![image](https://user-images.githubusercontent.com/68082808/99098323-d8fa1900-25a6-11eb-8295-97ea04eaf76c.png)

**Regression** is used to predict continuous variables. For instance, say that we're interested in predicting a person's weight based on factors such as height, diet, and exercise patterns. To accomplish this task, we would collect data on a number of people, the larger the pool the better. The regression model's algorithms would attempt to learn patterns that exist amongst these factors. If presented with the data of a new person, the model would make a prediction of that person's weight based on previously learned patterns from the dataset.

**Classification** on the other hand is used to predict discrete outcomes. For instance, say we're interested in using a person's traits, such as age, income, gender, sexual orientation, education level, and geographic location, to predict how they may vote on an issue. In this example the outcome is binary, either yes or no. The classification models algorithms would attempt to learn patterns from the data, and if the model is successful, gain the ability to make accurate predictions for new voters.

The difference between the two is in the outcome. A regression model uses parameters to predict the where an object might fit on a continuous graph. A classification model on the other hand uses parameters to predict which discrete group a new object may be in. In both models a dataset must first be split into features and target. Features are variables used to make a prediction. Target is the parameter we are trying to predict give the features.  
![image](https://user-images.githubusercontent.com/68082808/99126443-29d43680-25d4-11eb-8aac-8dfa4888e5dd.png)


### Linear Regression vs. Logistic Regression

Both models are commonly used in Data Science, however they do have their differences. Linear regression is continuous such that it can take any value within a specified range of values. For example, in [Scikit-learn demo](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/Scikit-learn%20demo.ipynb) we plot a linear regression model that predicts salary based on years of experience. The independent variable, YearsExperience in this case, is a continuous variable. On the otherhand, logistic regression deals with discrete variables. In [logistic regression](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/logistic_regression.ipynb) we generate some data into clusters and train the machine to predict which cluster a new data point may belong to given its parameters.

### Assess Accuracy, Precision, and Sensitivity

In [diabetes](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/diabetes.ipynb) we measured the performance of the logistic regression model with ```accuracy_score(y_test, y_pred)``` and found the model to be about 0.78% accurate when making predictions. Any given prediction falls under one of two categories: true or false. In the context of whether of not an individual as diabetes, a true prediction would mean that the model categorizes the individual as having diabetes. A false prediction means that the model categorizes the individual as not having diabetes. These are called binary classifications.

- If an individual is diagnosed positive, and they actually are positive, we call the prediction a true positive
- If an individual is diagnosed positive, and they are actually negative, we call the prediction a false positive
- If an individual is diagnosed negative, and they actually are positive, we call the prediction a false positive
- If an individual is diagnosed positive, and they are actually positive, we call the prediction a true negative

The following is a basic confusion matrix:  
![image](https://user-images.githubusercontent.com/68082808/99191067-3e314400-2738-11eb-8bcc-13adc0163ebe.png)

In machine learning, precision is a measure of how reliable a positive classification is. Precision is calculated as the ratio of true positives. Precision looks to see how much false positives got thrown in the mix.  
```Precision = TP/(TP + FP)```


Create training and test groups from a given data set.  
Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.  
Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.  
Compare the advantages and disadvantages of each supervised learning algorithm.  
Determine which supervised learning algorithm is best used for a given data set or scenario.  
Use ensemble and resampling techniques to improve model performance.  
