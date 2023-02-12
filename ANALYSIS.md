# Terminology

| Term                          | Notation | Meaning                                                         |
|-------------------------------|----------|-----------------------------------------------------------------|
| Input variable                | X        | Any given info based on which we must predict result            |
| Positive class probability    | P(y=1)   | Probability that client would subscribe a term deposit          |
| Linear feature                | -        | Feature equal to X                                              |
| Probability feature           | -        | P(y=1) for every unique value of an X                           |
| Linear probability feature    | -        | Inverse logistic function of probability feature                |
| Polynomial series of features | -        | Set of features, each representing some order of magnitude of X |

# Overview

![img.png](img/img.png)

```text
RangeIndex: 45211 entries, 0 to 45210
Total 17 columns
dtypes: int64(7), object(10)
```

**Target column**: y

# Closer look

### Age [0]

integer; non-null; [18, 95]

![img_1.png](img/img_1.png)

![img_23.png](img/img_23.png)

Numeric data. We'll derive a **polynomial series** of features from this column. From the graph we see that there are
very few examples with value > 80. This might lead to overfitting. We can't find more examples, so we'll try to filter
out them.

### Job [1]

text; non-null

![img_2.png](img/img_2.png)

![img_24.png](img/img_24.png)

This is categorical data. We'll derive **one-hot**, **probability** and **linear probability** features from this one to
see which one behaves better.

### Marital [2]

Marital status. Note: "divorced" means divorced or widowed.

text; non-null

![img_3.png](img/img_3.png)

![img_25.png](img/img_25.png)

This is categorical data. We'll derive **one-hot**, **probability** and **linear probability** features from this one to
see which one behaves better.

### Education [3]

text; non-null

![img_4.png](img/img_4.png)

![img_26.png](img/img_26.png)

This is categorical data. We'll derive **one-hot**, **probability** and **linear probability** features from this one to
see which one behaves better.

### Default [4]

Has credit in default?

text; non-null

![img_5.png](img/img_5.png)

![img_27.png](img/img_27.png)

This is binary data. The perfect case for **one-hot** encoding.

### Balance [5]

Average yearly balance, in euros

integer; non-null; [-8019, 102127]

![img_6.png](img/img_6.png)

![img_7.png](img/img_7.png)

![img_28.png](img/img_28.png)

Numeric data. We'll derive **polynomial series** of features from this one. Also, we have spikes of probability due to
lack of examples with value > 36039. This may lead to overfitting problem, so we'll try to filter out these examples.

### Housing [6]

Has housing loan?

text; non-null

![img_8.png](img/img_8.png)

![img_29.png](img/img_29.png)

This is binary data. The perfect case for **one-hot** encoding.

### Loan [7]

Has personal loan?

text; non-null

![img_9.png](img/img_9.png)

![img_30.png](img/img_30.png)

This is binary data. The perfect case for **one-hot** encoding.

### Contact [8]

Contact communication type. Related with the last contact of the current campaign.

text; non-null

![img_10.png](img/img_10.png)

![img_31.png](img/img_31.png)

This is categorical data. We'll derive **one-hot**, **probability** and **linear probability** features from this one to
see which one behaves better.

### Day [9]

Last contact day of the month. Related with the last contact of the current campaign.

integer; non-null; [1, 31]

![img_11.png](img/img_11.png)

![img_32.png](img/img_32.png)

Both numeric and categorical data. There might be a slight correlation between the size of the number and P(y=1). But
also it's likely that people tend to take biased decisions depending on specific dates. So we'll derive a **polynomial
series** of features, as well as **one-hot**, **probability** and **linear probability** features to see which one
behaves better.

### Month [10]

Last contact month of year. Related with the last contact of the current campaign.

text; non-null

![img_12.png](img/img_12.png)

![img_33.png](img/img_33.png)

Categorical data. We'll derive **one-hot**, **probability** and **linear probability** features from this one
to see which one behaves better.

### Duration [11]

Last contact duration, in seconds. Related with the last contact of the current campaign.

integer; non-null; [0, 4918]

![img_13.png](img/img_13.png)

![img_14.png](img/img_14.png)

![img_34.png](img/img_34.png)

Numeric data. We'll derive a **polynomial series** of features from this one.

### Campaign [12]

Number of contacts performed during this campaign and for this client (includes last contact).

integer; non-null; [1, 63]

![img_15.png](img/img_15.png)

![img_16.png](img/img_16.png)

![img_35.png](img/img_35.png)

Numeric data. Looks like a logistic function with a negative weight. We'll derive a **polynomial series** of features.
Also, notice that there are very few examples with value > 32. So, we'll try to filter them out to prevent overfitting.

### PDays [13]

Number of days that passed by after the client was last contacted from a previous campaign (-1 means client was not
previously contacted).

integer; non-null; [-1, 871]

![img_17.png](img/img_17.png)

![img_18.png](img/img_18.png)

![img_36.png](img/img_36.png)

Both numerical and categorical. We'll derive a **polynomial series** of features for this X. Also, it has a special
value of -1, which will add a bias to polynomial at this exact point. In order to prevent polynomial from distortion
we'll also introduce a **one-hot** variable for this special case. And finally, notice that there are very few examples
with value > 580. So, we'll try to filter them out to prevent overfitting.

### Previous [14]

Number of contacts performed before this campaign and for this client

integer; non-null; [0, 275]

![img_19.png](img/img_19.png)

![img_20.png](img/img_20.png)

![img_39.png](img/img_39.png)

![img_41.png](img/img_41.png)

![img_37.png](img/img_37.png)

![img_40.png](img/img_40.png)

Numeric data. We'll derive a **polynomial series** of features. Also, there are very few examples with value > 13. So,
we'll try to filter them out to prevent overfitting.

### POutcome [15]

Outcome of the previous marketing campaign.

text; non-null

![img_21.png](img/img_21.png)

![img_38.png](img/img_38.png)

This is categorical data. We'll derive **one-hot**, **probability** and **linear probability** features from this one to
see which one behaves better.

### Y [16]

Has the client subscribed a term deposit?

text; non-null

![img_22.png](img/img_22.png)

Classes will be: 1 (yes), 0 (no).

# Source

[Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of
the CRISP-DM Methodology.
In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121,
Guimarães, Portugal, October, 2011. EUROSIS.

Available at: [pdf] http://hdl.handle.net/1822/14838
[bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt