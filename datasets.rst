In-Built Datasets
=================
There are in-built datasets provided in both pandas and statsmodels packages.

** Statsmodels **

.. code:: python

  import statsmodels.api as sm
  prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data
  print prestige.head()

  
  type  income  education  prestige
  accountant  prof      62         86        82
  pilot       prof      72         76        83
  architect   prof      75         92        90
  author      prof      55         90        76
  chemist     prof      64         86        90

