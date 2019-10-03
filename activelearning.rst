Active Learning
================

Introduction
--------------
Getting labeled data is a huge and often prohibitive cost for a lot of machine learning projects.
Active Learning is a methodology that can sometimes greatly reduce the amount of labeled data required to train a model. 
It does this by prioritizing the labeling work for the experts (oracles).

Active Learning prioritizes which data the model is most confused about and requests labels for just those.
This helps the model learn faster, and lets the experts skip labeling data that wouldn’t be very helpful to the model.

There are generally 3 methods of sampling methods.

 * **Membership Query Synthesis**: a synethsized sample is sent to an oracle for labeling.
 * **Stream-Based Selective Sampling**: each sample is considered separately to be labeled or not. There are no assumptions on data distribution, and therefore it is adaptive to change.
 * **Pool-Based Sampling**: This is similar to stream-based, except that it starts a large pool of unlabelled data.



Resources
----------
 * Active Learning Literature Survey. http://burrsettles.com/pub/settles.activelearning.pdf