# Statistical Analysis R-to-Python Conversion Toolkit

## 🚀 Overview

This repository details a robust, end-to-end journey in building a modern, reusable statistical analysis toolkit—starting from collaborative R scripts, integrating all logic in R, translating to Python, and finally packaging everything into a Python class that fully replicates R’s workflow.

**The goal:**  
Seamlessly transition common R statistical workflows to Python while preserving analytical rigor, reproducibility, and ease of use.

## 📁 Project Structure and Steps

1. **R Scripts by Topic**  
   Each analytic technique was implemented as an independent R script with clear documentation, organized into folders:
   - `descriptive_statistics/`
   - `visualization/`
   - `assumption_checking/`
   - `test_visualization/`
   - ...and so on

2. **Unified R Function**  
   All topic folders were synthesized into a single, integrated R script/function, allowing a user to run an entire pipeline analysis with one function call for easier automation, reproducibility, and onboarding.

3. **Python Translation**  
   The end-to-end R workflow (and each statistical technique) was then translated into Python, using well-supported libraries for:
   - Data wrangling: `pandas`, `numpy`
   - Plots and visualizations: `matplotlib`, `seaborn`, `plotnine`
   - Statistical tests/models: `scipy.stats`, `statsmodels`, `sklearn.tree`
   - Heatmaps: `seaborn.clustermap`
   - Mixed models: `statsmodels.mixedlm`

4. **Python Statistical Class Blueprint**  
   All translated Python logic was encapsulated into a single, extensible Python class (`PyStatToolkit`), exposing R-style methods such as `.descriptive_stats()`, `.plot_histogram()`, `.anova()`, `.linear_mixed_model()`, etc.  
   The API is intentionally R-like for familiarity and ease of adoption.

## 🧭 Typical Workflow

1. **Write and test modular R scripts (one per analysis topic)**
   - Example: `descriptive_statistics/descriptive_statistics.R`

2. **Integrate scripts into a master R function**
   - Example: `run_complete_r_analysis()`

3. **Translate each R function into Python with best practices**
   - Example: `get_descriptive_stats(data)` ⇨ Python version

4. **Package all Python functions into a single class for use as a library**
   - Example usage:
     ```python
     from pystat_toolkit import PyStatToolkit
     toolkit = PyStatToolkit(my_dataframe)
     toolkit.descriptive_stats()
     toolkit.plot_histogram('score')
     toolkit.anova('score', 'group')
     toolkit.decision_tree('outcome', ['score', 'value1'])
     ```

## 🔑 Key Features

- **1:1 R to Python function mapping:**  
  Every essential R analysis can now be done in Python using a single unified class.

- **R-like syntax in Python:**  
  Methods and arguments closely mimic R (`.anova()`, `.kruskal_wallis()`, etc.)

- **Publication-ready plots:**  
  Histograms, boxplots, correlation matrices, heatmaps, raincloud plots, and decision trees.

- **Statistical rigor:**  
  Includes Wilcoxon, Kruskal–Wallis, ANOVA, mixed effects models, and more.

- **Extensible base class:**  
  New tests and plots can be easily added.

## 📚 Libraries Used

- **R:** `dplyr`, `ggplot2`, `corrplot`, `rpart`, `lme4`, `pheatmap`, `pastecs` etc.
- **Python:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`, `statsmodels`, `sklearn.tree`, `plotnine`

## 💾 How to Use (Python Example)

```python
import pandas as pd
from pystat_toolkit import PyStatToolkit

df = pd.read_csv("my_data.csv")
toolkit = PyStatToolkit(df)

toolkit.descriptive_stats()
toolkit.plot_histogram("score")
toolkit.plot_correlation_matrix()
toolkit.check_normality("score")
toolkit.anova("score", "group")
toolkit.decision_tree("outcome", ["score", "value1", "value2"])
toolkit.heatmap()
```

## Summary Table: R vs Python Functions

| R Task                   | R Function(s)            | Python Equivalent (`PyStatToolkit`)   | Library          |
|--------------------------|--------------------------|--------------------------------------|------------------|
| Descriptive statistics   | `summary()`              | `.descriptive_stats()`               | pandas           |
| Histogram                | `hist()`, `ggplot`       | `.plot_histogram()`                  | matplotlib, seaborn |
| Correlation plots        | `corrplot()`             | `.plot_correlation_matrix()`         | seaborn          |
| Q-Q plot/normality       | `qqnorm()`, `shapiro.test()` | `.check_normality()`               | scipy.stats      |
| Box/rain plots           | `boxplot()`, `ggplot`    | `.boxplot()`, `.violinplot()`        | seaborn          |
| Group comparisons        | various                  | `.group_comparison()`                | pandas, scipy    |
| Wilcoxon/Kruskal         | `wilcox.test()`, `kruskal.test()` | `.wilcoxon_test()`, `.kruskal_wallis()` | scipy      |
| ANOVA                    | `aov()`, `TukeyHSD()`    | `.anova()`                           | statsmodels      |
| Mixed effects            | `lmer()`                 | `.linear_mixed_model()`              | statsmodels      |
| Decision tree            | `rpart()`, `rpart.plot()`| `.decision_tree()`                   | scikit-learn     |
| Heatmaps                 | `pheatmap()`             | `.heatmap()`                         | seaborn          |




