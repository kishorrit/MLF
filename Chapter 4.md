# Chapter 4 - Time Series
Time series are the most iconic form of financial data. Virtually all media materials related to finance sooner or later show a stock price graph. Not a list of prices at a given moment, but a development of prices over time. Commenters frequently discuss the movement of prices ('Apple Inc. is up 5%, what does that mean?') but much less the absolute values ('A share of Apple Inc. is $137.74, what does that mean?'). This is because market participants are interested in how things will develop in the future and try to extrapolate from how things developed in the past. This is not only done in finance. Most forecasting involves looking at past developments over time. Farmers look at time series when forecasting crop yields for example. Because time series are so important to forecasting, a vast body of knowledge on working with them has developed. The fields of statistics, econometrics and engineering all have developed tools for working with and forecasting from time series. In this chapter, we will look at a few 'classic' tools that are still very much relevant today. We will then learn about how neural networks can deal with time series. Finally, we will have a look at how deep learning models can express uncertainty.

Many readers might have come to this chapter to read about stock market forecasting. This chapter is not about stock market forecasting and neither is any other chapter in this book. Economic theory shows, that markets are somewhat efficient. The efficient market hypothesis states that all publicly available information is included in stock prices. This extends to information on how to process information, such as forecasting algorithms. If this book were to present an algorithm that could predict prices on the stock market and deliver superior returns, many investors would implement this algorithm. Since these algorithms would all buy or sell in anticipation to price changes, they would change the prices in the present, thus destroying the advantage that the use of the algorithm would bring. Therefore, the algorithm presented would not work for future readers and they would learn less from the book. A solution to this problem would be to sell only one copy of the book for a few million dollars to a hedge-fund owner, much like Wu Tang Clan, but then again that did not go too well. Instead, this chapter uses traffic data from wikipedia. The goal is to forecast traffic to a specific wikipedia page. Wikipedia traffic data can be obtained via the `wikipediatrend` CRAN package. The dataset used here is traffic data of about 145 thousand wikipedia pages provided by Google. The data can be obtained from Kaggle, see the code appendix for download instructions.


# Visualization and preparation in pandas 
Data https://www.kaggle.com/c/web-traffic-time-series-forecasting

https://www.kaggle.com/muonneutrino/wikipedia-traffic-data-exploration

As we saw in chapter 2, it is usually a good idea to get an overview of the data before we start training. 

```Python 
train = pd.read_csv('../input/train_1.csv').fillna(0)
train.head()
```

| |Page|2015-07-01|2015-07-02|...|2016-12-31|
|-|----|----------|----------|----------|----------|
|0|2NE1_zh.wikipedia.org_all-access_spider|18.0|11.0|...|20.0|
|1|2PM_zh.wikipedia.org_all-access_spider|11.0|14.0|...|20.0|

The first item in a row contains the name of the page, the language of the wikipedia page, the type of accessing device and the accessing agent. The other columns contain the traffic for that page on that date. For example the first column is about the page of 2NE1, a Korean pop band, on the Chinese wikipedia version by all methods of access but only for agents classified as spider traffic, that is traffic not coming from humans. While most time series work is about local, time dependent features, we can enrich all of our models by providing access to _global features_. We therefore want to split up the page string into smaller, useful features.
```Python 
def parse_page(page):
    x = page.split('_')
    return ' '.join(x[:-3]), x[-3], x[-2], x[-1]
```
We split the string by underscores. Since the name of the page can include underscores, we join all elements up to the third last by a space to get the articles subject. The third last element is the sub url, (e.g. en.wikipedia.org). The second last element is the access and the last element the agent.
```Python 
parse_page(train.Page[0])
```
```
Out:
('2NE1', 'zh.wikipedia.org', 'all-access', 'spider')
```

When we apply this function to every page entry in the training set, we obtain a list of tuples which we can then join together into a new dataframe:

```Python 
l = list(train.Page.apply(parse_page))
df = pd.DataFrame(l)
df.columns = ['Subject','Sub_Page','Access','Agent']
```

Finally, we add this new dataframe back to our original dataframe and remove the original page column:

```Python 
train = pd.concat([train,df],axis=1)
del train['Page']
```

## Aggregate global feature statistics
After all this hard work, we can create some aggregate statistics on global features. Pandas `value_counts()` function allows us to easily plot the distribution of global features.

```Python 
train.Sub_Page.value_counts().plot(kind='bar')
```
![Sub_page_vcounts](./assets/wiki_sub_pages.png)

This plot shows the number of time series available for each sub page. Wikipedia has sub pages for different languages, and we can see that our dataset contains pages from the English (en), Japanese (ja), German (de), French (fr), Chinese (zh), Russian (ru) and Spanish (es) wikipedia. Both `commons.wikimedia.org` and `www.mediawiki.org` are used to host media files such as images. 

```Python 
train.Access.value_counts().plot(kind='bar')
```
![Access Method](./assets/wiki_access.png)

There are two possible access methods: Mobile or desktop. `all-access` seems to be aggregate statistics including both mobile and desktop access. 

```Python 
train.Agent.value_counts().plot(kind='bar')
```

![Agent](./assets/wiki_agents.png)

Similarly, there are time series only for spider agents and time series for all other access. 

In classic statistical modeling, the next step would be to analyze the effect of each of these global features and build models around them. However, this is not necessary if enough data and computing power is available. A neural network can discover the effects of the global features itself and create new features based on their interactions. There are only two real issues that need to be addressed for global features.

1. Is the distribution of features very skewed? If it is, then there might be only very few instances that possess a global feature and our model might overfit on this global feature. Imagine there were only very few articles from the Chinese wikipedia in the dataset. The algorithm might distinguish based on the feature too much then and overfit the few Chinese entries. Our distribution is relatively even, so we do not have to worry about this. 
2. Can features be easily encoded? Some global features can not be one hot encoded. Imagine we were given the full text of a wikipedia article with the time series. It would not be possible to use this feature straight away and some heavy preprocessing would have to be done to use it. In our case, there are relatively few, straight forward categories that can be one hot encoded. The subject names however can not be one hot encoded since there are too many of them.

## Examining sample time series

Next to examining global features, we have to look at a few sample time series to get an understanding of the challenge. In this section, we will plot the views for the english language page of Twenty One Pilots, a musical duo from the USA. We will plot the actual page views together with a ten day rolling mean.

```Python 
idx = 39457

window = 10


data = train.iloc[idx,0:-4]
name = train.iloc[idx,-4]
days = [r for r in range(data.shape[0] )]

fig, ax = plt.subplots(figsize=(10, 7))

plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title(name)

ax.plot(days,data.values,color='grey')
ax.plot(np.convolve(data, 
                    np.ones((window,))/window, 
                    mode='valid'),color='black')


ax.set_yscale('log')
```

There is a lot going on in this code snippet and it is worth going through it step by step. We first define which row we want to plot. The Twenty One Pilots article is row 39457 in the train dataset. We then define the window size for the rolling mean. We separate the page view data and the name from the overall dataset with Pandas `iloc` tool that allows us to index data by row, and column coordinates. Counting days rather than displaying all the dates of the measurements makes the plot easier to read, so we create a day counter for the X axis. Next, we set up the plot and make sure it has the desired size by setting `figsize`. We define the axis labels and the title. Now we first plot the actual page views. Our X coordinates are the days, and the Y coordinates are the actual page views. To compute the mean, we use a `convolve` operation. You might be familiar with convolutions from the third chapter. This convolve operation creates a vector of ones divided by the window size (10). The convolve operation slides the vector over the page view, multiplies ten page views with 1/10 and then sums the resulting vector up. This creates a rolling mean with window size 10. We plot this mean in black. Finally, we specify that we want to use a log scale for the Y axis.

![Twenty One Pilots](./assets/21_pilots_en.png)

You can see there are some pretty large spikes in the graph, even though we use a logarithmic axis. On some days, views skyrocket to 10X what they were the days before. It becomes clear, that a good model will have to be able to deal with such extreme spikes. It is also clearly visible that there are  global trends, as the page views generally increase over time. For good measure, lets plot the interest in Twenty One Pilots for all languages. 

```Python 
fig, ax = plt.subplots(figsize=(10, 7))
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Twenty One Pilots Popularity')
ax.set_yscale('log')

for country in ['de','en','es','fr','ru']:
    idx= np.where((train['Subject'] == 'Twenty One Pilots') 
                  & (train['Sub_Page'] == '{}.wikipedia.org'.format(country)) 
                  & (train['Access'] == 'all-access') & 
                  (train['Agent'] == 'all-agents'))
                  
    idx=idx[0][0]
    
    data = train.iloc[idx,0:-4]
    handle = ax.plot(days,data.values,label=country)
    

ax.legend()
```
In this snippet, we first set up the graph, as before. We then loop over the language codes and find the index of the Twenty One Pilots. The index is an array wrapped in a tuple so we have to extract the integer specifying the actual index. We then extract the page view data from the training dataset and plot the page views.

![All Countries](./assets/21_pilots_global.png)

There is clearly some correlation between the time series. The english language wikipedia is, not surprisingly by far the most popular. We can also see that the time series in our datasets are clearly not stationary, they change means and standard deviations. This undermines some of the assumptions many classic modeling approaches make. Yet, financial time series are hardly ever stationary, so it is worthwhile dealing with these problems and there are several good tools that can handle non stationarity.

# Fast Fourier transformations 
Another interesting statistic we often want to compute about time series is the Fourier transformation. Without going into the math, a Fourier transformation shows the amount of oscillation of a particular frequency in a function. You can imagine this like the tuner on an old FM radio. As you turn the tuner, you search through different frequencies. Every once in a while, you find a frequency that gives you a clear signal of a particular radio station. A Fourier transformation basically scans through the entire frequency spectrum and records at which frequencies there is a strong signal. This is useful for finding periodic patterns in time series data. Imagine that we found that the frequency 1/week gives a strong pattern. That would mean that knowledge about what the traffic was the same day last week would help our model. When both the function and the Fourier transform are discrete, which is the case in a series of daily measurements, it is called discrete Fourier transform (DFT). A very fast algorithm for computing DFT is called Fast Fourier Transform, which today has become an important algorithm in scientfific computing. It was know to Gauss in 1805 already, but brought to light by Cooley and Tukey. We will not go into how and why Fourier transformations work exactly mathematically, but only give a brief intuition. Imagine our function as a piece of wire. We take this wire, and wrap it around a point. If you wrap the wire so that the number of revolutions around the point matches the frequency of a signal, all the signal peaks will be on one side of the pole. This means that the center of mass of the wire will move away from the point we wrapped the wire around. In maths, wrapping a function around a point can be done by multiplying the function $g(n)$ with $e^{-2 \pi i f n}$. Where $f$ is the frequency of wrapping, $n$ is the number of the item from the series, $i$ is the imaginary square root of $-1$. Readers that are not familiar with imaginary numbers can think of them as coordinates in which each number has a two dimensional coordinate consisting out of a real and imaginary number. To compute the center of mass, we average the coordinates of the points in our discrete function. The DFT formula is therefore 

$$y[f] = \sum_{n=0}^{N-1} e^{-2 \pi i \frac{f n}{N}} x[n]$$

Where $y[f]$ is the fth element in the transformed series, or the frequency tested, and $x[n]$ is the nth element of the input series $x$. $N$ is the total number of points in the input series. Note that $y[f]$ will be a number with a real and a discrete element. To detect frequencies we are only really interested in the overall magnitude of $y[f]$ so we compute the root of the sum of the squares of the imaginary and real parts. 

In Python, we do not have to worry about all the math. We can use `sklearn`s `fftpack` which has a FFT function built in:

```Python 
data = train.iloc[:,0:-4]
fft_complex = fft(data)
fft_mag = [np.sqrt(np.real(x)*np.real(x)+
                   np.imag(x)*np.imag(x)) for x in fft_complex]
```

Here, we first extract the time series measurements without global features from our training set. We then run the FFT algorithm. Finally, we compute the magnitudes of the transformation. Now we have the Fourier transformations of all time series. We can average them to get a better insight in the general behavior.

```Python 
arr = np.array(fft_mag)
fft_mean = np.mean(arr,axis=0)
```

This first turns the magnitudes into a numpy array to then compute the mean. We want to compute the mean per frequency, not just the mean value of all magnitudes so we need to specify the `axis` along which to take the mean. In this case, the series are stacked in rows, so taking the mean column wise (axis zero) will result in frequency wise means. 

To better plot the transformation, we need to create a list of frequencies tested. The frequencies are day/all days in the dataset for each day, so 1/550, 2/550, 3/550, etc.

```Python 
fft_xvals = [day / fft_mean.shape[0] for day in range(fft_mean.shape[0])]
```

In this visualization we only care about the range of frequencies in a weekly range, so we will remove the second half of the transformation.

```Python 
npts = len(fft_xvals) // 2 + 1
fft_mean = fft_mean[:npts]
fft_xvals = fft_xvals[:npts]
```

And now we finally get to plot our transformation! 

```Python 
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(fft_xvals[1:],fft_mean[1:])
plt.axvline(x=1./7,color='red',alpha=0.3)
plt.axvline(x=2./7,color='red',alpha=0.3)
plt.axvline(x=3./7,color='red',alpha=0.3)
```

![Global FFT](./assets/global_fft.png)

There are spikes at 1/week, speak information from a week ago helps, 2/week, information from half a week ago and 3/week, information from a third of a week ago. The spikes have already been marked with red lines. 

# Autocorrelation

Autocorrelation is the correlation between two elements of a series separated by a given interval. Intuitively, we would for example assume that knowledge about the last time step helps us forecasting the next step. But how about knowledge from two time-steps ago or from 100 steps ago? Autocorrelation plots help answer these questions. An `autocorrelation_plot` plots the correlation between elements with different lag times.

Pandas comes with a handy autocorrelation plotting tool. To use it, we have to pass a series of data. In this case, we pass the page views of a page, selected at random.

```Python 
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(data.iloc[110])
plt.title(' '.join(train.loc[110,['Subject', 'Sub_Page']]))
```

![Single Autocorr](./assets/single_autocorr.png)

The plot shows the correlation of page views for the wikipedia page of 'Oh My Girl', a South-Korean girl group, in the Chinese wikipedia. You can see that shorter time intervals between 1 and 20 days show a higher autocorrelation than longer intervals. But there are also curious spikes, such as around 120 days and 280 days. Annual, quarterly or monthly events could lead to frequent visits to a wikipedia page. We can examine the general pattern of these frequencies by drawing 1000 of these autocorrelation plots.

```Python 
a = np.random.choice(data.shape[0],1000)

for i in a:
    autocorrelation_plot(data.iloc[i])
    
plt.title('1K Autocorrelations')
```

This code snippet first samples 1000 random numbers between zero and the number of series in our dataset (about 145K). We use these as indices to randomly sample rows from our dataset for which we then draw the autocorrelation plot.

![Autocorrelation 1K](./assets/1K_autocorrelations.png)

First, we see that autocorrelations can be quite different for different series and that there is a lot of noise. There seems to be a general trend towards higher correlations around 350 days, or roughly annual. It makes sense to incorporate annual lagged page views as a time dependent feature as well as the autocorrelation for one year time intervals as a global feature. The same is true for quarterly and half year lag as these seem to have high autocorrelations or sometimes quite negative autocorrelations which makes them valuable as well. 

Time series analysis like the examples shown above help engineer features for our model. Complex neural networks could in theory discover all features by themselves, however, it is often much easier to help them a bit, especially with information about long time periods. 

# Establishing a training & testing regime
Even with lots of data available, we have to ask ourselves how we want to split data into training, validation and testing data. The dataset already comes with a test set of future data, so we do not have to worry about the test set. For the validation set, there are two ways of splitting: 

![Time series splits](./assets/time_series_splits.png)

In a walk forward split, we train on all 145 thousand series. To validate, we use more recent data from all series. In side by side splitting we sample a number of series for training and use the rest for validation. Both have advantages and disadvantages. The disadvantage of walk forward splitting is that we can not use all observations of the series for our predictions. The disadvantage of side by side splitting is that we can not use all series for training. If we have few series, but many data observations per series, a walk forward split is preferable. It also aligns more nicely with the forecasting problem, at hand. In side by side splitting, the model might overfit to global events in the prediction period. Imagine that wikipedia was down for a week in the prediction period used in side by side splitting. This would reduce views for all pages, and the model would overfit to this global event. We would not catch the overfitting in our validation set as the prediction period in our validation set is also affected by the global event. However, in our case we have many time series, but only about 550 observations per series. There seem to be no global events that would have significantly impacted all wikipedia pages in a time period. However, there are some global events that impacted views for some pages, such as the olympic winter games. Yet, this is a reasonable risk in this case, as the number of pages affected by such global events are still small. Since we have an abundance of series and only few observations per series, a side by side split is more feasible in our case.

In this chapter we focus on forecasting traffic for 50 days. So we first split the last 50 days of each series form the rest before splitting training and validation set.

```Python
from sklearn.model_selection import train_test_split

X = data.iloc[:,:500]
y = data.iloc[:,500:]

X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, 
                                                  test_size=0.1, 
                                                  random_state=42)
``` 
When splitting, we use `X.values` to only get the data, not a DataFrame containing the data. This operations leaves us with 130,556 series to train and 14,507 for validation. We use mean absolute percentage (MAPE) error as a loss and evaluation metric. MAPE can cause division by zero errors if the true value of y is zero. We thus use a small value epsilon to prevent division by zero.

```Python 
def mape(y_true,y_pred):
    eps = 1
    err = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    return err
```

# A note on backtesting
The peculiarities of choosing training and testing sets are especially important in systematic investing and algorithmic trading. The main way to test trading algorithms is a process called **backtesting**. Backtesting means we train the algorithm on data from a certain time period and then test its performance on _older_ data. For example we could train on data from 2015 to 2018 and then test on data from 1990 to 2015. Usually not only the models accuracy is tested, but the backtested algorithm executes virtual trades so its profitability can be evaluated. Backtesting is done because there is plenty of past data available. 

Backtesting suffers from several biases such as
- Look-Ahead bias: Introduced if future data is accidentally included at a point in the simulation where that data would not have been available yet. This can be caused by a technical bug in the simulator. But it can also stem from parameter calculation. If a strategy makes use of the correlation between two securities for example and the correlation is calculated for all time once, a look-ahead bias is introduced. The same goes for the calculation of maxima or minima.
- Survivorship bias: Introduced if only stocks that still exist at the time of testing are included in the simulation. Consider for example the 2008 financial crisis in which many firms went bankrupt. Leaving the stocks of these firms out when building a simulator in 2018 would introduce survivorship bias. After all, the algorithm could have invested in those stocks in 2008. 
- Psychological tolerance bias: What looks good in a backtest might not be good in real life. Consider an algorithm that looses money for 4 months in a row before making it all back in a backtest. We might feel satisfied with this algorithm. However, if the algorithm looses money for 4 months in a row in real life and we do not know if it makes it back, will we sit tight or pull the plug? In the backtest we know the final result, but in real life we do not.
- Overfitting: A problem for all machine learning algorithms but in backtesting, overfitting is a persistent and insidious problem. Because not only does the algorithm potentially overfit, the designer of the algorithm might use knowledge about the past to and build an algorithm that overfits to it. It is easy to pick stocks in hindsight and the knowledge can be incorporated in models which then look great in backtests. It might be subtle, such as relying on certain correlations that held up well in the past but it is easily to unconsciously build bias into models that are evaluated in backtesting.

Building good testing regimes is a core activity of any quantitative investment firm or anyone working intensively with forecasting. Two popular strategies to test algorithms other than backtesting are testing models on data that is statistically similar to stock data but generated. We might build a generator for data that looks like real stock data but is not real, thus avoiding knowledge about real market events creeping into our models. Another option is to deploy models silently and test them in the future. The algorithm runs but executes only virtual trades so that if things go wrong, no money is lost. This approach makes use of future data instead of past data. The downside is that we have to wait for quite a while until the algorithm can be used.

In practice, a combination regimes is used. Statisticians carefully design regimes to see how an algorithm responds to different simulations. In our web traffic forecasting model we will simply validate on different pages and then test on future data in the end.

# Median forecasting
A good sanity check and sometimes underrated forecasting tool are medians. A median is the value separating the higher half of a distribution from the lower half, they sit exactly in the middle of the distribution. Medians have the advantage of removing noise. They are less susceptible to outliers than means and capture the mid point of a distribution. They are also easy to compute. 

To make a forecast, we compute the median over a look-back window in our training data. In this case, we use a window size of 50, but you could experiment with other values. We then select the last 50 values from our X values and compute the median. Note that in the numpy median function, we have to set `keepdims=True`. This ensures that we keep a two dimensional matrix rather than a flat array which is important when computing the error. 

```Python 
lookback = 50

lb_data = X_train[:,-lookback:]

med = np.median(lb_data,axis=1,keepdims=True)

err = mape(y_train,med)
``` 

We obtain an error of about 68.1%, not bad given the simplicity of our method. To see how the medians work, lets plot the X values, true y values and predictions for a random page.

```Python 
idx = 15000

fig, ax = plt.subplots(figsize=(10, 7))


ax.plot(np.arange(500),X_train[idx], label='X')
ax.plot(np.arange(500,550),y_train[idx],label='True')

ax.plot(np.arange(500,550),np.repeat(med[idx],50),label='Forecast')

plt.title(' '.join(train.loc[idx,['Subject', 'Sub_Page']]))
ax.legend()
ax.set_yscale('log')
```  

As you can see, our plotting consists of drawing three plots. For each plot we specify the X and Y values for the plot. For `X_train`, the X values range from zero to 500, for `y_train` and the forecast they range from 500 to 550. From our training data, we select the series we want to plot by indexing. Since we have only one median value, we repeat the median forecast of the desired series 50 times to draw our forecast. 

![Median Forecast](./assets/median_forecast.png)

As you can see, the data for this page, the image of American actor Eric Stoltz is very noisy, and the median cuts through all the noise. This is especially useful for pages that are visited infrequently and for which there is no clear trend or pattern. 

A lot of further work could be done with medians. You could for example use different medians for weekends or use a median of medians from multiple look-back periods. A simple tool like median forecasting can deliver good results with smart feature engineering. It makes sense to use a bit of time on implementing it as a baseline and sanity check before using more advanced methods.

# ARIMA
https://www.kaggle.com/zoupet/predictive-analysis-with-different-approaches

# Kalman filters
http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/README.md

# Time series neural nets

# Conv1D

# SimpleRNN
Another method to make order matter in neural networks is to give the network some kind of memory. So far, all of our networks did a forward pass without any memory of what happened before or after the pass. It is time to change that with recurrent neural networks.

![Simple RNN Cell](./assets/simple_rnn.png)

Reocurrent neural networks contain reocurrent layers. Reocurrent layers can remember their last activation and use it as their own input.

$$A_{t} = activation( W * in + U * A_{t-1} + b)$$

A reocurrent layer takes a sequence as an input. For each element, it then computes a matrix multiplication ($W * in$) just like a ``Dense`` layer and runs the result through an activation function like e.g. ``relu``. It then retains it's own activation. When the next item of the sequence arrives, it performs the matrix multiplication as before but it also multiplies it's previous activation with a second matrix ($U * A_{t-1}$). It adds the result of both operations together and passes it through it's activation function again. In Keras, we can use a simple RNN like this:

# LSTM 

In the last section we already learned about basic recurrent neural networks. In theory, simple RNN's should be able to retain even long term memories. However, in practice, this approach often falls short. This is because of the 'vanishing gradients' problem. Over many timesteps, the network has a hard time keeping up meaningful gradients. See e.g. Learning long-term dependencies with gradient descent is difficult (Bengio, Simard and Frasconi, 1994) for details.

In direct response to the vanishing gradients problem of simple RNN's, the Long Short Term Memory layer was invented. Before we dive into details, let's look at a simple RNN 'unrolled' over time:

![Unrolled RNN](./assets/unrolled_simple_rnn.png)

You can see that this is the same as the RNN we saw in the previous chapter, just unrolled over time.

## The Carry 
The central addition of an LSTM over an RNN is the carry. The carry is like a conveyor belt which runs along the RNN layer. At each time step, the carry is fed into the RNN layer. The new carry gets computed in a separate operation from the RNN layer itself from the input, RNN output and old carry.

![LSTM](./assets/LSTM.png)

The ``Compute Carry`` can be understood as three parts:

Determine what should be added from input and state:

$$i_t = a(s_t \cdot Ui + in_t \cdot Wi + bi)$$

$$k_t = a(s_t \cdot Uk + in_t \cdot Wk + bk)$$

where $s_t$ is the state at time $t$ (output of the simple rnn layer), $in_t$ is the input at time $t$ and $Ui$, $Wi$ $Uk$, $Wk$ are model parameters (matrices) which will be learned. $a()$ is an activation function.

Determine what should be forgotten from state an input:

$$f_t = a(s_t \cdot Uf) + in_t \cdot Wf + bf)$$

The new carry is the computed as 

$$c_{t+1} = c_t * f_t + i_t * k_t$$

While the standard theory claims that the LSTM layer learns what to add and what to forget, in practice nobody knows what really happens inside an LSTM. However, they have been shown to be quite effective at learning long term memory.

Note that ``LSTM``layers do not need an extra activation function as they already come with a tanh activation function out of the box.
# Recurrent dropout 

# Bigger time series models
Conv1D + GRU


# Uncertainty in neural nets - Bayesian deep learning
http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html

https://github.com/kyle-dorman/bayesian-neural-network-blogpost/blob/master/README.md