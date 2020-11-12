# Supervised_ML_Credit_Risk

In this project we use Python to build and evaluate a number of machine learning models to predict credit risk. The ability to predict credit risk with machine learning algorithms helps banks and other financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud. In this project we will use machine learning to predict credit risk, this may provide a quicker and more reliable loan experience. Machine learning may also provide insight on good candidates for loans, which may lead to lower default rates. We will build and evaluate several machine learning models/algorithms to predict credit risks. Resampling and boosting are some of the techniques we will use to make the most of our models and our data.

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

Explain how a machine learning algorithm is used in data analytics.
Create training and test groups from a given data set.
Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.
Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.
Compare the advantages and disadvantages of each supervised learning algorithm.
Determine which supervised learning algorithm is best used for a given data set or scenario.
Use ensemble and resampling techniques to improve model performance.
