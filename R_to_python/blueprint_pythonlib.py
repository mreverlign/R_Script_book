import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.tree import DecisionTreeClassifier, plot_tree

class PyStatToolkit:
    def __init__(self, data):
        self.data = data

    def descriptive_stats(self):
        return self.data.describe(include='all')

    def plot_histogram(self, col):
        sns.histplot(self.data[col], kde=True)
        plt.show()

    def plot_correlation_matrix(self):
        corr = self.data.corr()
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
        plt.show()

    def check_normality(self, col):
        stats.probplot(self.data[col], dist="norm", plot=plt)
        plt.show()
        stat, p = stats.shapiro(self.data[col].dropna())
        return {'Shapiro-Wilk W': stat, 'p-value': p}

    def boxplot(self, x, y):
        sns.boxplot(x=self.data[x], y=self.data[y])
        plt.show()

    def violinplot(self, x, y):
        sns.violinplot(x=self.data[x], y=self.data[y])
        plt.show()

    def group_comparison(self, x, y):
        groups = self.data.groupby(x)[y]
        return groups.describe()

    def wilcoxon_test(self, col1, col2):
        return stats.wilcoxon(self.data[col1], self.data[col2])

    def kruskal_wallis(self, y, group):
        groups = [g[y].dropna() for name, g in self.data.groupby(group)]
        return stats.kruskal(*groups)

    def anova(self, y, x):
        model = smf.ols(f"{y} ~ C({x})", data=self.data).fit()
        return sm.stats.anova_lm(model, typ=2)

    def linear_mixed_model(self, y, x, group):
        model = smf.mixedlm(f"{y} ~ {x}", self.data, groups=self.data[group])
        return model.fit().summary()

    def decision_tree(self, target, features):
        X = self.data[features]
        y = self.data[target]
        tree = DecisionTreeClassifier().fit(X, y)
        plt.figure(figsize=(16,8))
        plot_tree(tree, feature_names=features, class_names=np.unique(y).astype(str), filled=True)
        plt.show()
        return tree

    def heatmap(self, cols=None):
        if cols is not None:
            data = self.data[cols]
        else:
            data = self.data.select_dtypes(include=np.number)
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        plt.show()
