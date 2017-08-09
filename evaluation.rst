Evaluation
==========

Accuarcy is widely used for evalution, but others like, user satisfaction, increased in patient survival, 
etc., can be used too depending on circumstance.


Confusion Matrix
----------------

.. figure:: images/confusion.png
    :width: 300px
    :align: center

    Wikipedia

.. figure:: images/confusion2.png
    :width: 400px
    :align: center

    Wikipedia
        
**Sensitivity|Recall**: True Positive / True Positive + False Negative. High recall means to get all 
true positives despite having some false positives.
Search & extraction in legal cases, Tumour detection. Often need humans to filter false positives.

**Precision**: True Positive / True Positive + True Negative. High precision means it is important 
to filter off the any false positives.
Search query suggestion, Document classification, customer-facing tasks. 

**F1-Score**: is the harmonic mean of precision and sensitivity

**1. Confusion Matrix**

Plain vanilla matrix.

>>> print sklearn.metrics.confusion_matrix(test_target,predictions)
[[14  0  0]
 [ 0 13  0]
 [ 0  1 10]]

Using a heatmap.

.. code:: python
  
   # create confusion matrix
   confusion_mc = confusion_matrix(y_test_mc, svm_predicted_mc)
   # convert to a dataframe
   df_cm = pd.DataFrame(confusion_mc, 
                        index = [i for i in range(0,10)], 
                        columns = [i for i in range(0,10)])
   # plot graph
   plt.figure(figsize=(5.5,4)) # define graph
   sns.heatmap(df_cm, annot=True) # draw heatmap, add annotation


.. image:: images/confusion3.png
    :scale: 50 %
    :align: center


**2. Evaluation Metrics**

.. code:: python

  from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
  
  # Accuracy = TP + TN / (TP + TN + FP + FN)
  # Precision = TP / (TP + FP)
  # Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
  # F1 = 2 * Precision * Recall / (Precision + Recall) 
  
  print('Accuracy:', accuracy_score(y_test, tree_predicted)
  print('Precision:', precision_score(y_test, tree_predicted)
  print('Recall:', recall_score(y_test, tree_predicted)
  print('F1:', f1_score(y_test, tree_predicted)
  
  Accuracy: 0.95
  Precision: 0.79
  Recall: 0.60
  F1: 0.68


**3. Classification Report**

.. code:: python

  # Combined report with all above metrics
  from sklearn.metrics import classification_report

  print(classification_report(y_test, tree_predicted, target_names=['not 1', '1']))
  
                precision    recall  f1-score   support

        not 1       0.96      0.98      0.97       407
            1       0.79      0.60      0.68        43

  avg / total       0.94      0.95      0.94       450
  
  
Precision-Recall Curves
-----------------------

ROC Curves
----------

Receiver Operating Characteristic (ROC) is used to show the performance of a binary classifier. 
Area Under Curve (AUC) of a ROC is used 
