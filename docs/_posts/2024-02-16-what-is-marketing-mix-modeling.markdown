---
layout: single
title:  "What is Marketing Mix Modeling (MMM)?"
date:   2024-02-16 00:00:00 -2300
categories: Marketing
---

# What is marketing mix modeling?

Marketing mix modeling is a modeling technique designed to describe what marketing factors influence key business success metrics. If we want to gauge how advertising campaigns, products offerings, or pricing strategies each affect our business's sales, then a marketing mix model would allow us to evaluate which efforts have the highest impact on our bottom line.

# How is this model created?

Typically, marketing mix models are formulated using **regression models**, though you can use any model that describes the influence of one set of variables over another, like an influence maximization graph model or an agent-based simulation model. I talk about regression models [here]({% post_url 2024-02-01-an-introduction-to-regression %}).

By creating a line of best fit between the marketing variabels we have influence over and our goal metric like net sales, revenue, profit, etc., we are able to create an extension of a simple linear regression called a multi-linear regression model. Rather than using only one independent variable to predict, this model uses multiple features to predict. 

# Why is it useful?

Just like with any regression model, we would be able to see which variables influence our target variable the most. We can use this model to observe causal relationships and see which areas of marketing to focus on to achieve the highest ROI. Not only can we infer which course of actions are most useful for our goals, we can use this model to forecast into the future and see how much revenue or profit we can achieve by simulating shifts in marketing focus.

# An example with Python's sklearn and Pytorch

To showcase an example, we use data from an [Advertising Sales Dataset found on Kaggle](https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset), where we use ad budgets and sales values to capture relationships between the two areas. 

{% highlight ruby%}
## Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

## Read in data
df = pd.read_csv('Advertising Budget and Sales.csv')
df.head(5)
{% endhighlight %}
We now see a snapshot of the data: ![DataSample]({{ site.url }}{{ site.baseurl }}/assets/images/MMM_01.png)

{% highlight ruby%}
## Drop unnecessary columns
df1 = df.drop(columns = ['Unnamed: 0'])

## Split data into X  as input and y as output for our future models
X = df1.iloc[:, :-1]
y = df1.iloc[:, -1:]

from sklearn.model_selection import train_test_split

## Separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8)

## Visualize correlation map 
ax = sns.heatmap(X.corr(), annot = True)
plt.show()
{% endhighlight %}
We use a correlation map to check for any collinearity amongst our features. Low correlation values means that it is more likely that a linear relationship exists between our features and our target values of sales. 

![CorrelationMap]({{ site.url }}{{ site.baseurl }}/assets/images/MMM_02.png)

Below, we run both a linear regression model and neural network regression model. We decide to use sklearn and Pytorch respectively for our libraries of choice. 
{% highlight ruby%}
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,explained_variance_score, PredictionErrorDisplay

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print('Regression coefficients: ', reg.coef_)
print('Regression Intercept: ', reg.intercept_)
print('RMSE: ', np.sqrt(mean_squared_error(y_pred, y_test)))
print('MAE: ', mean_absolute_error(y_pred, y_test))
print('Explained Variancce: ', explained_variance_score(y_pred, y_test))

residual = y_pred - y_test
plt.hist(residual, bins=25)
plt.xlabel('Prediction Error')
plt.show()
{% endhighlight %}

![LinearRegressionResults]({{ site.url }}{{ site.baseurl }}/assets/images/MMM_05.png)

From the regression coefficients, we can see conclude that since the second coefficient is the highest, the second feature in the dataset, the **radio ad budget**, has the most effect on the sales.

{% highlight ruby%}
## Import libraries 
import torch.nn as nn
import torch.optim as optim
import torch
import tqdm
import copy

## Define neural network. Here, we only use one hidden layer.
model = nn.Sequential(
    nn.Linear(3, 12),
    nn.ReLU(),
    nn.Linear(12, 1),
)

## Define loss and optimizers.
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)

## Convert data into tensors.
X_train_tensor = torch.tensor(X_train.values, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32).reshape(-1, 1)

## Define epochs and batch sizes.
n_epochs = 500
batch_size = 5
batch_start = torch.arange(0, len(X_train_tensor), batch_size)

## Create variables to store results.
best_rmse = np.inf
best_weights = None
history = []

## Start training!
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval = 0, disable = True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            X_batch = X_train_tensor[start:start + batch_size]
            y_batch = y_train_tensor[start:start + batch_size]
            
            y_pred = model(X_batch)
            loss = torch.sqrt(loss_fn(y_pred, y_batch))
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            bar.set_postfix(rmse=float(loss))
    model.eval()
    y_pred = model(X_test_tensor)
    rmse = torch.sqrt(loss_fn(y_pred, y_test_tensor))
    rmse = float(rmse)
    history.append(rmse)
    if rmse < best_rmse:
        best_rmse = rmse
        best_weights = copy.deepcopy(model.state_dict())

## Load and test on the best model        
model.load_state_dict(best_weights)
y_pred = model(X_test_tensor)

print("RMSE: %.2f" % best_rmse)
plt.plot(history)
plt.show()
{% endhighlight %}
From our result, we ended up getting an RMSE of 1.44, with the following loss curve: ![LossCurve]({{ site.url }}{{ site.baseurl }}/assets/images/MMM_03.png)

We plot the residuals to ensure it follows a normal distribution so that we satisfy the assumptions of our linear regression model, confirming that linear relationship does in fact exist within the data.
{% highlight ruby%}
## Plot residuals
error = y_pred.tolist() - y_test.values
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
plt.show()
{% endhighlight %}
![LossCurve]({{ site.url }}{{ site.baseurl }}/assets/images/MMM_04.png)

Since the residuals of both models do follow a normal distribution, along with our low RMSE, we can be satisfied with using a linear regression model for our data.

With the RMSE from both models being relatively close, we can assume both models perform similarly. The RMSE of around 1 applied to our dataset indicates that the average error of our predicted sales value from the true sales value is typcally around $1 mil. 

# Conclusion

Marketing mix models are an effective tool in deriving relationships between marketing metrics and financial impact metrics. Using these models can help showcase the causal relationships between marketing variables and KPIs, which business leads can use to determine which marketing leads to shift focus to. Once these models are built, they can also forecast future outcomes based on user input to gauge what is the expected financial gain or loss from adjusting focus on different marketing streams. Marketing mix models are simple to build yet effective in quantifying marketing strategy impact. 




