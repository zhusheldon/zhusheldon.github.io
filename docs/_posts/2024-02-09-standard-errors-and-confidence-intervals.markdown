---
layout: single
title:  "How Important are Standard Errors and Confidence Intervals?"
date:   2024-02-09 00:00:00 -2300
categories: Statistics
mathjax: true
---

# What is standard error?

It is best to understand standard error by first understanding what sampling is. When we are collecting data on a **population**, then we must sample from this population. **Sampling** is simply collecting data (or samples) that reflect the group or population that you aim to observe. **Random sampling** occurs when each sample is considered **independent** from one another, meaning that the action of choosing one sample does not influence the next sample. When random sampling is performed, it is assumed that samples are pulled each time from the same underlying distribution (i.e. the true population distribution).

After performing random sampling, you will obtain statistics based off of the data you sample, including the sample mean and sample standard deviation. However, how can we evaluate how far off these sample statistics are from the true value, like the true mean or true standard deviation? We use **standard error** which is a single positive value, telling us how far or close we can usually expect the true statistic to be from the sample statistic. The standard error value is a building block for the confidence interval. 

# What is a confidence interval?

Say other people repeat this data collection procedure, and they obtain their own sample statistics. Given a certain probability value $$p$$ (a.k.a. the **confidence level**), their sample statistic, and the standard error, then they can then construct a **confidence interval**. This is a range centered around the sample statistic which usually increases and decreases with the confidence level. If people continuously collect sets of data on this population, we will obtain a collection of confidence intervals, and we can infer that $p$ percent of the confidence intervals contain the true population statistic. 

# An Example

To better visualize this, let's simulate this sampling process from a standard normal distribution. We define $$X_1$$ as our set of data taken from a standard normal distribution, while $$X_2$$ is simply $$X_1$$ duplicated twice. $$X_2$$ is created to show how increasing the sample size decreases the standard error and the confidence interval range. While not shown below, standard error is proportional to standard deviation of the data, and so increasing standard deviation would increase both standard error and the confidence interval.

{% highlight ruby %}
## Import libraries
import scipy
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
np.random.seed(1)

## Define number of samples n
n = 25

## Create distributions x1 and x2
X1 = np.random.normal(0, 1, n)
X2 = np.concatenate([X1, X1])

## Calculate sample mean, standard error, and confidence intervals
m1, m2 = np.mean(X1), np.mean(X2)
sem1, sem2 = scipy.stats.sem(X1), scipy.stats.sem(X2)
me1, me2 = sem1 * scipy.stats.norm.ppf(1-  0.05/ 2.), sem2 * scipy.stats.norm.ppf(1-  0.05/ 2.)
print(f"Mean of x1: {m1:.2f}")
print(f"Mean of x2: {m2:.2f}")
print(f"Standard error of mean of x1: {sem1:.4f}")
print(f"Standard error of mean of x2: {sem2:.4f}")
print(f"CI of mean of x1: ({m1 - me1:.2f}, {m1+me1:.2f})")
print(f"CI of mean of x2: ({m2 - me2:.2f}, {m2+me2:.2f})")
{% endhighlight %}
Output:
<br>
Mean of X1: -0.01<br>
Mean of X2: -0.01<br>
Standard error of mean of X1: 0.2196<br>
Standard error of mean of X2: 0.1537<br>
CI of mean of X1: (-0.44, 0.42)<br>
CI of mean of X2: (-0.31, 0.29)<br>

Let us see how the confidence intervals are visualized.
{% highlight ruby %}
sns.set_style('whitegrid')
ax = sns.kdeplot(np.array(X1), bw=0.5)
ax.axvline(x = np.mean(X1), ymin = 0, ymax = 1, color = 'red', label = 'Data')
ax.axvspan(m1 - me1,  m1 + me1, alpha = 0.2, color = 'green',)
ax.set_title(f'Distribution of x1 (n = {n})')
{% endhighlight %}
![Distribution of X1]({{ site.url }}{{ site.baseurl }}/assets/images/x1stdnorm.png)
{% highlight ruby %}
sns.set_style('whitegrid')
ax = sns.kdeplot(np.array(X2), bw=0.5)
ax.axvline(x = np.mean(X2), ymin = 0, ymax = 1, color = 'red', label = 'Data')
ax.axvspan(m2 - me2,  m2 + me2, alpha = 0.2, color = 'green',)
ax.set_title(f'Distribution of x2 (n = {n*2})')
{% endhighlight %}
![Distribution of X2]({{ site.url }}{{ site.baseurl }}/assets/images/x2stdnorm.png)

Notice that the confidence interval of $X_2$ is smaller than the confidence interval of $X_1$.

# How important are confidence intervals?

Confidence intervals give us a range of values that we can interpret as possible values for our statistic. When dealing with sampling, we must recognize that our sample statistic is not actually the true statistic value, but is instead reasonably close to what we want. In our case above, we see that our sample mean of $-0.01$ is pretty close to the true mean of $0$, and the confidence interval is able to capture that in this case as well. If we didn't know that the true mean is $0$, we can say that any value within this confidence interval can be justified as a true population mean that is supported by our samples. 

Of course, there is always a chance that the true population statistic is not captured within the confidence interval. With our confidence level set at $95\%$, we are saying that if we repeatedly ran the code above, $95\%$ of our confidence intervals will capture the true mean of $0$. The above confidence interval could have very well not contained $0$. To minimize this sort of error occuring that can lead to false assumptions without doing additional sampling, we can increase the confidence level to something like $99\%$, giving us a wider confidence interval but safer estimates on where the true mean will be contained. 

Therefore, we come to one of the common struggles with confidence intervals. We can judge whether our result is significant from our single group of data and extrapolate it over a series of additional experiments, but we cannot determine whether our confidence interval truly contains the true statistic or not. Whether this confidence interval can help us find our desired statistic is a fault within frequentist statistics, but exploring Bayesian statistics can solve this issue. We will discuss this in a another blog post.

# Conclusion

Standard error and confidence intervals give us a way to evaluate our data samples and gives us more information than our single-valued sample statistic. Confidence intervals grow with confidence level and standard deviation and decreases as we increase our sample size. While confidence intervals give a range of possible values for our true statistic, it doesn't actually give us any value with certainty, which is one of the common criticisms of the confidence interval and frequentist statistics in general, which assumes you have infinite chances to sample from your population. Thus, while still offering reliable inference, we want to be wary when using confidence intervals to make conclusions, as, like all forms of statistics, they are not designed to provide 100% certainty. So when looking at confidence intervals, always ask youself, "What is the statistic being measured, and how reliable is this interval?"





