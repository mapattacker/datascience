Distance Measure
================

Euclidean Distance & Cosine Similarity
--------------------

Euclidean distance is the straight line distance between points, while cosine distance is the cosine of the angle
between these two points.

.. figure:: images/distance1.png
  :width: 400px
  :align: center


.. code:: python

    from scipy.spatial.distance import euclidean

    euclidean([1,2],[1,3])
    # 1
    

.. code:: python
    
    from scipy.spatial.distance import cosine

    cosine([1,2],[1,3])
    # 0.010050506338833642


Mahalanobis Distance
---------------------

.. code:: python

    from scipy.spatial.distance import mahalanobis


Dynamic Time Warping
---------------------

If two time series are identical, but one is shifted slightly along the time axis, 
then Euclidean distance may consider them to be very different from each other. 
DTW was introduced to overcome this limitation and give intuitive distance measurements 
between time series by ignoring both global and local shifts in the time dimension.

DTW is a technique that finds the optimal alignment between two time series, 
if one time series may be “warped” non-linearly by stretching or shrinking it along its time axis.
Dynamic time warping is often used in speech recognition to determine if two waveforms 
represent the same spoken phrase. In a speech waveform, 
the duration of each spoken sound and the interval between sounds are permitted to vary, 
but the overall speech waveforms must be similar.

From the creators of FastDTW, it produces an accurate minimum-distance warp path between two time series than is nearly optimal 
(standard DTW is optimal, but has a quadratic time and space complexity).

Output: Identical = 0, Difference > 0

.. code:: python

    import numpy as np
    from scipy.spatial.distance import euclidean
    from fastdtw import fastdtw

    x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
    y = np.array([[2,2], [3,3], [4,4]])
    distance, path = fastdtw(x, y, dist=euclidean)
    print(distance)

    # 2.8284271247461903

Stan Salvador & Philip ChanFast. DTW: Toward Accurate Dynamic Time Warping in Linear Time and Space. Florida Institude of Technology. https://cs.fit.edu/~pkc/papers/tdm04.pdf