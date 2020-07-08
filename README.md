Machine Learning based suicide prediction has the potential to improve our understanding of suicide while saving lives, it raises many risks that have been underexplored. 

Dataset
In our problem, the data that should be feeded for the machine to decide and predict effectively has to be measure of variability in depressive symptoms along with other relevant factors such as younger age, mood disorders, childhood abuse, and personal and parental history of suicide attempts, etc.

Columns in csv file containing overview from 1965–2016:
Before loading the dataset into our code, we first import the necessary libraries.
Numpy is a package in Python used for Scientific Computing ; matplotlib.pyplot is a plotting library used for 2D graphics ; Pandas is the most popular python library that is used for data analysis.
Next, we import the dataset.

The dataset is in form of a csv file containing data in the above mentioned columns,
Our model follows Supervised Learning, which consists in learning the link between two datasets: the observed data X and an external variable y that we are trying to predict, usually called “target” or “labels”. Most often, y is a 1D array of length n_samples.

All supervised estimators in scikit-learn implement a fit(X, y) method to fit the model and a predict(X) method that, given unlabeled observations X, returns the predicted labels y.

While assigning values to X, we drop some columns which we do not require or which are less relevant to our model while predicting the output.

Investigating Correlation
Correlation is a technique for investigating the relationship between two quantitative, continuous variables, for example, age, sex and number of suicides.

This involves investigating the connection between the scatterplot of bivariate data and the numerical value of the correlation coefficient.

We observe that no two variables are linearly correlated.
We are considering both as some countries like the Soviet Union had a high GDP per capita but did not distribute the wealth.
Checking for Outliers
Many machine learning algorithms are sensitive to the range and distribution of attribute values in the input data. Outliers in input data can skew and mislead the training process of machine learning algorithms resulting in longer training times, less accurate models and ultimately poorer results.

We check for outliers in the input labels and data by plotting scatter plots of the columns.

Columns in the dataset
We observe there are outliers above suicide rates of 125 and over based on GDP and HDI.
Since we observe there are outliers above suicide rates of 125 and over based on GDP and HDI, it is preferable to drop them.

Data Preprocessing
Steps in Data Preprocessing
Data is preprocessed as per the model deployed. The generalized preprocessing we do initially is as follows.

We replace the commas from the values for the data to be converted as float.
A machine learning pipeline is used to help automate machine learning workflows. They operate by enabling a sequence of data to be transformed and correlated together in a model that can be tested and evaluated to achieve an outcome, whether positive or negative.

Below, we pipeline steps to fill in missing values with the mean of the values, scale and normalize the values and encode the values using One Hot Encoding.

The rest of the dataset is preprocessed as required by the respective models deployed.

Splitting the Dataset
As we work with datasets, a machine learning algorithm works in two stages. We have split the data around 20%-80% between testing and training stages.

Under supervised learning, we split a dataset into a training data and test data in Python ML.
Trying Linear Regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output).

Performance Evaluation
There are various metrics that can be used to evaluate the performance of a Linear Regression model. We will use the RMSE (Root Mean Squared Error) value, which is a frequently used measure of the differences between values (sample and population values) predicted by a model and the values actually observed.

Trying Support Vector Regression
A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples.

Performance Evaluation
Trying Decision Tree Regression
Decision Tree is a decision-making tool that uses a flowchart-like tree structure or is a model of decisions and all of their possible results, including outcomes, input costs and utility.

Decision-tree algorithm falls under the category of supervised learning algorithms. It works for both continuous as well as categorical output variables.We can see that if the maximum depth of the tree (controlled by the max_depth parameter) is set too high, the decision trees learn too fine details of the training data and learn from the noise, i.e. they overfit.

Performance Evaluation
Trying Random Forest Regression
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees’ habit of overfitting to their training set.

Performance Evaluation
Comparison of Results by Different Algorithms
Let us take a look at the collected RMSE values by different algorithms used.

As it is evidently visible, the RMSE value is the least for the Random Forest Regression for the testing set and quite an optimal value for the training set as well, thus, it performs the best for our model.

