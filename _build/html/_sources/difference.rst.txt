Tests of Difference
===================

Chi-Square Test
---------------
X, Explantory: ``Categorical``
Y, Response: ``Categorical``
Type: ``Non-Parametric``

.. code:: python

  print 'chi-square statistic, p-value, expected counts'
  print ss.chi2_contingency(ct1)
  
  chi-square statistic, p-value, expected counts
  (1263.6306705804054, 2.554837585615145e-272, 4, array([[  7.74251477e+03,   1.71950205e+03,   3.69930718e+02,
            4.25495413e+01,   2.50291420e+00],
         [  7.72448523e+03,   1.71549795e+03,   3.69069282e+02,
            4.24504587e+01,   2.49708580e+00]]))


Student's T-Test
----------------
Type: ``Parametric``


ANOVA
-----
Type: ``Parametric``

Analysis of Variance (ANOVA). X, or independent variable can be continuous or categorical. 
Can input mutiple factors (x).

**Categorical Independent Variable**

.. code:: python

  #### IMPORT MOUDLES ####
  import numpy as np
  import pandas as pd
  import statsmodels.formula.api as smf
  import statsmodels.stats.multicomp as multi



  #### FIT MODEL ####
  # response~explanatory OR x~y, 'C' refers to categorical variable
  # ANOVA for multiple factors
  model = smf.ols(formula='diameter ~ C(layers)', data=df3)
  results = model.fit()
  >>> print results.summary()


  OLS Regression Results                            
  ==============================================================================
  Dep. Variable:               diameter   R-squared:                       0.219
  Model:                            OLS   Adj. R-squared:                  0.219
  Method:                 Least Squares   F-statistic:                     1383.
  Date:                Tue, 02 Aug 2016   Prob (F-statistic):               0.00
  Time:                        17:04:57   Log-Likelihood:                -60976.
  No. Observations:               19731   AIC:                         1.220e+05
  Df Residuals:                   19726   BIC:                         1.220e+05
  Df Model:                           4                                         
  Covariance Type:            nonrobust                                         
  ==================================================================================
  coef    std err          t      P>|t|      [95.0% Conf. Int.]
  ----------------------------------------------------------------------------------
  Intercept          6.7217      0.043    157.125      0.000         6.638     6.806
  C(layers)[T.2]     3.3941      0.100     33.822      0.000         3.197     3.591
  C(layers)[T.3]    12.2841      0.200     61.319      0.000        11.891    12.677
  C(layers)[T.4]    18.3139      0.579     31.649      0.000        17.180    19.448
  C(layers)[T.5]    21.8123      2.380      9.166      0.000        17.148    26.477
  ==============================================================================
  Omnibus:                    14916.319   Durbin-Watson:                   0.529
  Prob(Omnibus):                  0.000   Jarque-Bera (JB):           577157.627
  Skew:                           3.262   Prob(JB):                         0.00
  Kurtosis:                      28.680   Cond. No.                         64.0
  ==============================================================================

  Warnings:
  [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




  #### POST-HOC TEST ####
  mc = multi.MultiComparison(df3['diameter'],df3['layers'])
  result1 = mc.tukeyhsd()
  print result1
  
  
  Multiple Comparison of Means - Tukey HSD,FWER=0.05
  =============================================
  group1 group2 meandiff  lower   upper  reject
  ---------------------------------------------
    1      2     3.3941   3.1204  3.6679  True 
    1      3    12.2841  11.7376 12.8306  True 
    1      4    18.3139  16.7353 19.8925  True 
    1      5    21.8123  15.3204 28.3041  True 
    2      3      8.89    8.3015  9.4785  True 
    2      4    14.9198  13.3262 16.5134  True 
    2      5    18.4181  11.9226 24.9137  True 
    3      4     6.0298   4.3675  7.6921  True 
    3      5     9.5281   3.0154 16.0409  True 
    4      5     3.4984  -3.1806 10.1773 False 
  ---------------------------------------------
  
**Multiple Factors**

.. code:: python

  reg2 = smf.ols('depth~diameter_new+layers+lat+long',data=df3).fit()
  print reg2.summary()
  
  
  
  OLS Regression Results                            
  ==============================================================================
  Dep. Variable:                  depth   R-squared:                       0.518
  Model:                            OLS   Adj. R-squared:                  0.518
  Method:                 Least Squares   F-statistic:                     4856.
  Date:                Wed, 03 Aug 2016   Prob (F-statistic):               0.00
  Time:                        17:58:20   Log-Likelihood:                -1393.4
  No. Observations:               18067   AIC:                             2797.
  Df Residuals:                   18062   BIC:                             2836.
  Df Model:                           4                                         
  Covariance Type:            nonrobust                                         
  ================================================================================
  coef    std err          t      P>|t|      [95.0% Conf. Int.]
  --------------------------------------------------------------------------------
  Intercept        0.5112      0.005     94.795      0.000         0.501     0.522
  diameter_new     0.0438      0.000    122.136      0.000         0.043     0.045
  layers           0.0061      0.004      1.555      0.120        -0.002     0.014
  lat             -0.0007    5.3e-05    -13.557      0.000        -0.001    -0.001
  long             0.0001   1.93e-05      7.131      0.000      9.97e-05     0.000
  ==============================================================================
  Omnibus:                      598.651   Durbin-Watson:                   1.078
  Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1210.720
  Skew:                          -0.234   Prob(JB):                    1.25e-263
  Kurtosis:                       4.179   Cond. No.                         342.
  ==============================================================================

  Warnings:
  [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.