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

### Pre-Process Data

Data preprocessing is a process of preparing raw data and making it suitable for a machine learning model. It is the first step while creating a machine learning model. Real-world data is often noisy, missing values, and perhaps in an unusable format. Thankfully for us there are libraries we can use that can help us prepare the data. Here is the general process:

1. Encode the data that is in the wrong format
2. Decide what to do with columns missing values
3. Define the features and target
4. Split into training and testing sets
5. Create a StandardScalar instance
6. Fit the StandardScalar
7. Scale the data

Your process might generally look like [this](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/Random-Forests.ipynb):  
```
# Define features set
X = df_loans.copy()
X = X.drop("bad", axis=1)
X.head()

# Define target vector
y = df_loans["bad"].values.reshape(-1, 1)
y[:5]

# Splitting into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Creating StandardScaler instance
scaler = StandardScaler()

# Fitting Standard Scaller
X_scaler = scaler.fit(X_train)

# Scaling data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

### Encode Labels with Pandas

While many datasets contain categorical features (e.g., M or F), machine learning algorithms typically only work with numerical data. Categorical and text data must therefore be converted to numerical data for use in machine learning. [Here](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/categorical-data.ipynb) we are originally given a data set with columns containing categorical data, using the following code we are able to quantify the values:  
```
# Binary encoding using Pandas
loans_binary_encoded = pd.get_dummies(loans_df, columns=["education", "gender"])
loans_binary_encoded = pd.get_dummies(loans_df, columns=["education", "gender"])
loans_df["month_num"] = loans_df["month"].apply(lambda x: months_num[x])
```  
This is an important step of the proccess, for instance to use Scikit-learn's machine learning algorithms, the text features (month, education, and gender) will have to be converted into numbers. The ```pd.get_dummies()``` method takes in two arguments, the first one specifies the dataframe, and the second on specifies the column(s). 

### Scale and Normalize Data

Data scaling and normalization are steps that are sometimes necessary when preparing data for machine learning. Earlier, we worked with a dataset that had already been scaled for us: the values in each column were rescaled to be between 0 and 1. Such scaling is often necessary with models that are sensitive to large numerical values. This is normally the case with models that measure distances between data points. SVM is one model that usually benefits from scaling.  
```loans_data_scaled = data_scaler.fit_transform(encoded_df)```

### Linear Regression vs. Logistic Regression

Both models are commonly used in Data Science, however they do have their differences. Linear regression is continuous such that it can take any value within a specified range of values. For example, in [Scikit-learn demo](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/Scikit-learn%20demo.ipynb) we plot a linear regression model that predicts salary based on years of experience. The independent variable, YearsExperience in this case, is a continuous variable. On the otherhand, logistic regression deals with discrete variables. In [logistic regression](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/logistic_regression.ipynb) we generate some data into clusters and train the machine to predict which cluster a new data point may belong to given its parameters.

### Assess Accuracy, Precision, and Sensitivity

In [diabetes](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/diabetes.ipynb) we measured the performance of the logistic regression model with ```accuracy_score(y_test, y_pred)``` and found the model to be about 0.78% accurate when making predictions. Any given prediction falls under one of two categories: true or false. In the context of whether of not an individual as diabetes, a true prediction would mean that the model categorizes the individual as having diabetes. A false prediction means that the model categorizes the individual as not having diabetes. These are called binary classifications.

- If an individual is diagnosed positive, and they actually are positive, we call the prediction a **true positive**
- If an individual is diagnosed positive, and they are actually negative, we call the prediction a **false positive**
- If an individual is diagnosed negative, and they actually are positive, we call the prediction a **false positive**
- If an individual is diagnosed positive, and they are actually positive, we call the prediction a **true negative**

The following is a basic confusion matrix:  
![image](https://user-images.githubusercontent.com/68082808/99191523-31fab600-273b-11eb-805a-21a06859cc20.png)

In machine learning, **precision** is a measure of how reliable a positive classification is. Precision is calculated as the ratio of true positives. Precision looks to see how the number false positives got thrown in the mix. Ask yourself, "given I was tested positive for diabetes, how likely is it that I have diabetes?"  
```Precision = TP/(TP + FP)```

Another way to assess a model's performance is with **sensitivity**, also called recall. While the term recall is more commonly used in machine learning, the two terms are synonymous and will often be used interchangeably. Instead of looking at the number of false positives the model predicted, recall looks at the number of false negatives that were thrown into the prediction mix. Ask yourself, "given I know I have diabetes, how likely is it that the test will diagnose me as positive?"  
```Sensitivity = TP/(TP + FN)```

Given our example of testing for diabetes, sensitivity is more important than precision! A test with high sensitivity means few false negatives, though there may be a high number of false positives. In this context, false positives are preferable to false negatives. It’s better to rule out false positive diagnoses than to miss patients who actually have diabetes. Why is high sensitivity more important than precision for a diabetes screening test? It's better to detect everyone who might have diabetes, even if it means a certain number of false positives, than to miss people who do have diabetes. After all, those with a positive result for diabetes can undergo further testing to confirm or rule out diabetes. The false positives in a highly sensitive test are accepted as a cost of doing business. This is not the general rule, the context determines which of the two, precision or sensitivity, is more important. 

In summary, there's a fundamental tension between precision and sensitivity. Highly sensitive tests and algorithms tend to be aggressive, as they do a good job of detecting the intended targets, but also risk resulting in a number of false positives. High precision, on the other hand, is usually the result of a conservative process, so that predicted positives are likely true positives; but a number of other true positives may not be predicted. In practice, there is a trade-off between sensitivity and precision that requires a balancing act between the two.

The **F1 score**, also called the harmonic mean, can be characterized as a single summary statistic of precision and sensitivity. The formula for the F1 score is the following: ```2(Precision * Sensitivity)/(Precision + Sensitivity)```  The F1 score is 0.002. We noted previously that there's usually a trade-off between sensitivity and precision, and that a balance must be struck between the two. A useful way to think about the F1 score is that a pronounced imbalance between sensitivity and precision will yield a low F1 score.

![image](https://user-images.githubusercontent.com/68082808/99191487-fb24a000-273a-11eb-8425-770d827c1c29.png)

### Support Vector Machine (SVM)

Support vector machine (SVM), like logistic regression, is a binary classifier: It can categorize samples into one of two categories (for example, yes or no). SVM categorizes the target variable into one of two classes (for example, approved or denied). However, it differs from logistic regression in several ways. As a linear classifier, the goal of SVM is to find a line that separates the data into two classes. The margins, however, are soft and can make exceptions for outliers. This stands in contrast to the logistic regression model. In logistic regression, any data point whose probability of belonging to one class exceeds the cutoff point belongs to that class; all other data points belong to the other class. An example of running an SVM model in Python can be found [here](https://github.com/sfnxboy/Supervised_ML_Credit_Risk/blob/main/demo/svm_loan_approver.ipynb).

### Random Forests

The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model. In the example below, there are multiple algorithms that are being used to learn and make their predictions based on the data. The final prediction is based on the accumulated predictions from each algorithm:  
![image](https://user-images.githubusercontent.com/68082808/99197897-4c488a00-2763-11eb-801e-2bb3bdee8c56.png)  
a random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features:  
![image](https://user-images.githubusercontent.com/68082808/99197944-87e35400-2763-11eb-8245-a6870b288edf.png)

Random forest algorithms are beneficial because they:  
- Are robust against overfitting as all of those weak learners are trained on different pieces of the data.
- Can be used to rank the importance of input variables in a natural way.
- Can handle thousands of input variables without variable deletion.
- Are robust to outliers and nonlinear data.
- Run efficiently on large datasets.
