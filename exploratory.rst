Exploratory Data Analysis
=========================

Exploratory data analysis (EDA) is an essential step to understand the data better;
in order to engineer and select features before modelling.
This often requires skills in visualisation to better interpret the data.


Box Plots
----------
Using the 50 percentile to compare among different classes, it is easy to find feature that
can have high prediction importance if they do not overlap. Also can be use for outlier detection.
**Features have to be continuous.**

.. code:: python

    plt.figure(figsize=(7, 5))
    cmap = sns.color_palette("Set3")
    sns.boxplot(x='Cover_Type', y='Elevation', data=df, palette=cmap);
    plt.xticks(rotation=45);

.. image:: images/box1.png
    :scale: 50 %
    :align: center

Multiple Plots

.. code:: python

    cmap = sns.color_palette("Set2")

    fig, axes = plt.subplots(ncols=2, nrows=5, figsize=(10, 18))
    a = [i for i in axes for i in i]
    for i, ax in enumerate(a):
        sns.boxplot(x='Cover_Type', y=eda2.columns[i], data=eda, palette=cmap, width=0.5, ax=ax);

    # rotate x-axis for every single plot
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)

    # set spacing for every subplot, else x-axis will be covered
    plt.tight_layout()


Correlation Plots
------------------

Heat-Map
*********
Heatmaps show a quick overall correlation between features.

Using plot.ly

.. code:: python

    from plotly.offline import iplot
    from plotly.offline import init_notebook_mode
    import plotly.graph_objs as go
    init_notebook_mode(connected=True)

    # create correlation in dataframe
    corr = df[df.columns[1:]].corr()

    layout = go.Layout(width=1000, height=600, \
                    title='Correlation Plot', \
                    font=dict(size=10))
    data = go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns)
    fig = go.Figure(data=[data], layout=layout)
    iplot(fig)

.. image:: images/corr1.png
    :scale: 60 %
    :align: center

Using seaborn

.. code:: python

    import seaborn as sns
    import matplotlib.pyplot as plt
    %config InlineBackend.figure_format = 'retina'
    %matplotlib inline

    # create correlation in dataframe
    corr = df[df.columns[1:]].corr()

    plt.figure(figsize=(15, 8))
    sns.heatmap(corr, cmap=sns.color_palette("RdBu_r", 20));


.. image:: images/corr2.png
    :scale: 60 %
    :align: center