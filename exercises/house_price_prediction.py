# from IMLearn.utils import split_train_test
# from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
from IMLearn.learners.regressors import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt

from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    #
    # sample_data = [[1, 2, 'a'], [3, 4, 'b'], [5, 6, 'c'], [7, 8, 'b']]
    # df = pd.DataFrame(sample_data, columns=['numeric1', 'numeric2', 'categorical'])
    # X = df[['categorical']]
    # print(df)
    # dummies = pd.get_dummies(data=X, drop_first=True)
    # df = df.join(dummies)
    # print(df)
    # return

    # data = pd.DataFrame({'color': ['blue', 'green', 'green', 'red']})
    # print(data)
    # print(pd.get_dummies(data))
    # return
    design = pd.read_csv(filename)
    response = design.pop("price")
    # print(response)
    # for col in design.columns:
    #     print(col, design[col])
    # design = pd.get_dummies(design, columns=['zipcode'])
    design.drop(columns=['lat', 'long', 'zipcode', 'date', 'id'], inplace=True)
    return design, response
    # print(design)
    # print(response)
    # print(design.columns)
    # design.drop("price", axis=1, inplace=True)
    # print(response)
    # print("\nColumns Names")

    # print("\nDf train shape")

    # print(design.shape)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for x in X:
        # print(x)
        if x in {'date', 'id'}:
            continue
        pearson = (np.cov(X[x], y) / (np.std(X[x]) * np.std(y)))[0][0]
        # print(pearson)
        plt.title(f"Price as a function of {x}, $\\rho$={pearson}")
        plt.xlabel(f"{x}")
        plt.ylabel('Price')
        plt.scatter(X[x], y)
        plt.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    size_loss = np.ndarray((91, 2))
    for p in range(10, 101):
        size = int(np.floor(train_X.shape[0] * p / 100))
        loss = np.ndarray(10)
        for i in range(10):
            lr = LinearRegression()
            tx = train_X[:size]
            ty = train_y[:size]
            lr.fit(tx, ty)
            loss[i] = lr.loss(test_X, test_y)
        size_loss[p - 10][0] = loss.mean()
        size_loss[p - 10][1] = loss.var()
    plt.title("MSE values as function of p% - log scale")
    plt.xlabel('Percentage (p%)')
    plt.ylabel('log MSE')
    plt.yscale("log")
    plt.plot(np.linspace(10, 101, 91), size_loss[:, 0])
    plt.show()
    print(size_loss)
    # fig = go.Figure([go.Scatter(x=np.linspace(10, 100, 1), y=size_loss[:, 0],
    #                             name="average loss as function of training size", showlegend=True,
    #                             marker=dict(color="black", opacity=.7),
    #                             line=dict(color="black", dash="dash", width=1))],
    #                 layout=go.Layout(title=r"$\text{(1) Simulated Data}$",
    #                                  xaxis={"title": "x - Explanatory Variable"},
    #                                  yaxis={"title": "y - Response"},
    #                                  height=400))
    # fig.show()
