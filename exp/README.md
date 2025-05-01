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

## plot_styleguide.ipynb

Was originally meant as a styleguide, but for now merely using `plt.style.use('seaborn-v0_8')` is good enough. Any future experimentations with plots will be done in this notebook.
