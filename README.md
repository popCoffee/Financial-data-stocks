# Financial-data-stocks
Data visualization of multiple bank stocks with python. Then an application of Tensorflow in python to create a neural network to predict the closing price of the given stocks.

## Analysis

```
The first plot represents all of the stocks. The second once is BAC's rolling average 
and BAC closing price which is used as the input to the neural network.
```

![alt text](https://github.com/popCoffee/Financial-data-stocks/blob/master/pics/price_day.png)
![alt text](https://github.com/popCoffee/Financial-data-stocks/blob/master/pics/BAC.png)

```
Distribution of a bank stock representing the range of the possible returns.
```
![alt text](https://github.com/popCoffee/Financial-data-stocks/blob/master/pics/distPlot_MS_.jpg)

```
A feedforward model for the neural network used. Not to scale.
```
![alt text](https://github.com/popCoffee/Financial-data-stocks/blob/master/pics/feedforward1.jpg)

```
A plot of the error from the neural network.
```
![alt text](https://github.com/popCoffee/Financial-data-stocks/blob/master/pics/LearnCurve.png)

## Final Output 
The final output of the neural network. It closely matched the original curve when predicting each subsequent point.
![alt text](https://github.com/popCoffee/Financial-data-stocks/blob/master/pics/final_nn.png)
