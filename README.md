# N-Step Time Series Predictor


### What is time series ?
A **time series** is a series of [data points](http://www.wikizero.biz/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvRGF0YV9wb2ludA "Data point") indexed (or listed or graphed) in time order. Most commonly, a time series is a [sequence](http://www.wikizero.biz/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvU2VxdWVuY2U "Sequence") taken at successive equally spaced points in time. Thus it is a sequence of [discrete-time](http://www.wikizero.biz/index.php?q=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvRGlzY3JldGUtdGltZQ "Discrete-time") data.

for more information: [https://en.wikipedia.org/wiki/Time_series](https://en.wikipedia.org/wiki/Time_series "Time Series")

### Why should I use this project ?
With this project you can find the most suitable MLP model for your dataset in the range of number of neurons, input indices and target indices you have chosen and save it for later uses.  
In addition you can learn the values of n-step ahead of your time series dataset and the autocorrelation coefficient between the input values and target values you choosed (work just when the input indices number and the target indices number is equal).

### Example of use:
You know that the selling rates of the last two days are affect the next day selling rates then you can set the input indices to 1,2 and the terget to 3 then train the model with your dataset, after that set the first values to the last two days selling rates and make the model predict the next days selling rates for a month.


**Note:**
The predicted values are saved as csv files.

<img src="/screenshots/ss(4).png" width="250"/>



### Usage: 

	$ git clone https://github.com/0xihsn/N-StepPredictor.git
	$ cd N-StepPredictor

For installing the required libraries

	$ pip install -r requirements.txt

And it's ready:

	$ python N-StepPredictor.py
The supported dataset format is:
````
month;data
1949-01;-213
1949-02;-564
1949-03;-35
1949-04;-15
1949-05;141
1949-06;115
````
 

## Screenshots:

<img src="/screenshots/ss(1).png" width="250"/>
<img src="/screenshots/ss(2).png" width="250"/>
<img src="/screenshots/ss(3).png" width="250"/>




