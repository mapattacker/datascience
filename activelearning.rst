Active Learning
================

The key idea behind active learning is that a machine learning algorithm can achieve 
greater accuracy with fewer training labels if it is allowed to choose the data from which it learns. 
An active learner may pose queries, usually in the form of unlabeled data instances 
to be labeled by an oracle (e.g., a human annotator).

Sampling Method > Query Strategy > Oracle

There are generally 3 methods of sampling methods.
 * **Membership Query Synthesis**: learner generates an instance, and is sent to oracle to label.
 * **Stream-Based Selective Sampling**: 
 * **Pool-Based Sampling**: Pool-based sampling is a type of stream-based selective sampling, with the difference that it has already a large sample of unlabelled data. The best instance is 

Resources
----------
 * Active Learning Literature Survey. http://burrsettles.com/pub/settles.activelearning.pdf