# `/exp` folder

This folder only contains files that are used for data exploration, and generally useful files that will not be part of the main data pipeline. Files that are part of the data pipeline are only saved in `/src`.

## Pairs Trading using Deep Learning.ipynb

(including PT DL copy.ipynb)
This file is the code that was uploaded as resources for the paper "Quantitative Trading Strategies Using Deep Learning: Pairs Trading (LSTMs for Pairs Trading)" by Kaur at Stanford.

The original code can be found here: https://github.com/sk3391/CS230/blob/master/Pairs%20Trading%20using%20Deep%20Learning.ipynb.

## train_transformer.ipynb

Go through this file to understand more about the thought process behind specific choices made when training the Transformer. It uses many of the modules.

## train_transformer_correct.ipynb

In this file, you can see specific steps that were taken to find out how the problem of using future data in z-score can be fixed.

## train_kalman_filter.ipynb

See more in detail the thought process and results for Kalman Filters.

## plot_styleguide.ipynb

Was originally meant as a styleguide, but for now merely using `plt.style.use('seaborn-v0_8')` is good enough. Any future experimentations with plots will be done in this notebook.

## gather_data.ipynb

Often, I came across the problem of having to scrape the yfinance data for a large set of ETFs every time I started programming. This notebook solves that problem by helping me in caching all the scraped data. If you want to save any specific new dataset, simply use the gather_data.ipynb with your wanted params.

## train_time_moe.ipynb

Technically this is used for finetuning a pre-trained model, but the name train_ was used to keep in line with existing naming conventions for other notebooks.