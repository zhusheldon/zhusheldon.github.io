---
layout: single
title:  "An Introduction to Regression (An example in Python)"
date:   2024-02-01 00:00:00 -0500
categories: Statistics
---

# What is the purpose of regression?

Regression is a modeling technique that describes what kind of relationship one **target variable** shares with one or more other **features**. Typically, this target variable is a continuous numerical value. One reason to build out regression models is to **predict** or **forecast** how the target value changes when we change the values of one of the features. Another reason is to evaluate how influential each feature is in affecting the target. Determining how strong the **causal relationship** is between a certain feature and the target will allow us to focus more on it in the future.

![Regression Example]({{ site.url }}{{ site.baseurl }}/assets/images/regression_heightvsweight.png)
* A example of regression showcasing the linear relationship between people's height and weight.

# How can we use regression?

When we have some data and we want to see what type of relationship exists between different features, then we can use regression. There are many different types of regression to determine what type of relationship variables share. The most simple type of regression is **simple linear regression**, which draws a line between two variables. This line helps us try and predict future target values when we only know one of the values.

# An example in Python

We can easily generate our own linear regression model using the Python programming language.
{% highlight ruby %}
## Import  libraries
import scipy
import numpy as np
import matplotlib.pyplot as plt 

## Generate feature variable x and target variable y
np.random.seed(0)
x = np.random.uniform(0, 50, 100)
var = 3
y = x + np.random.normal(0, var, len(x))

{% endhighlight %}

Plotted (x,y) data points:
{% highlight ruby %}
plt.scatter(x, y)
plt.show()
{% endhighlight %}
![Scatter Plot]({{ site.url }}{{ site.baseurl }}/assets/images/samplescatter.png)

We generate the *x* feature variable as values taken randomly between 0 and 50, while the *y* target variable is the same as its corresponding *x* value with additional noise added to it.

We are now able to generate our linear regression model, and we finish by plotting our data compared to the regression line.
{% highlight ruby %}
import statsmodels.regression.linear_model as linear_model
import statsmodels.api as sm
## Preprocessing: Add constant term. If we don't have this, then y will always equal 0 when x=0.
X = sm.add_constant(x)

## Generate linear regression model
model  = linear_model.OLS(y, X, hasconst = True)
results = model.fit()

## Show scatter plot with linear regression line
plt.plot(X[:, 1], results.predict(X), c = 'red', label = 'Predicted values')
plt.scatter(x, y, label = 'Actual Values', c = 'blue')
plt.legend()
plt.show()
{% endhighlight %}
![Linear Regression]({{ site.url }}{{ site.baseurl }}/assets/images/samplelinearregression.png)

This type of relationship we see between the two variables is likely something we've all seen before. Although the scale in our example above can be adjusted, some real-life examples that model this behavior include 

* Exercise Time vs. Calories Burned
* Education Level vs. Income
* Service Workers' Paycheck Amount vs. Hours Worked
* Student's Sleep Duration vs. Academic Performance
* Product Price vs. Value to Consumer 

# Common misuses of regression

Regression models are not designed to provide 100% accuracy, nor may they even be correct depending on the types of features and data we are dealing with. There are assumptions in the underlying data we must satisfy to a reasonable degree to ensure that regression models provide us **reliable** results. These assumptions require more statistical knowledge, but once we understand these, we can go forward with applying perhaps more complex regression models and more **applicable** predictions.

# Conclusion

We can see how regression, and in particular simple linear regression, can provide us a best-fit model for our underlying data. This is helpful for us to tease out underlying relationships within the data that we may want to predict in the future. Check out other blog posts to explore more in-depth findings on regression, including different types of regression models for non-linear data, the data assumptions for reliable linear regression, and the mathematical reasoning behind regression model fitting.
