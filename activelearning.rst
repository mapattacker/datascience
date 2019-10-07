Active Learning
================

Introduction
--------------
Getting labeled data is a huge and often prohibitive cost for a lot of machine learning projects.
Active Learning is a methodology that can sometimes greatly reduce the amount of labeled data required to train a model
with higher accuracy if it is allowed to choose which data to label. 
It does this by prioritizing the labeling work for the experts (oracles).

Active Learning prioritizes which data the model is most confused about and requests labels for just those.
This helps the model learn faster, and lets the experts skip labeling data that wouldn’t be very helpful to the model.

Sampling Methods
-----------------

1) **Membership Query Synthesis**: a synethsized sample is sent to an oracle for labeling.

2) **Stream-Based Selective Sampling**: each sample is considered separately to be labeled or not. There are no assumptions on data distribution, and therefore it is adaptive to change.

3) **Pool-Based Sampling**: This is similar to stream-based, except that it starts a large pool of unlabelled data.

The main difference between stream-based and pool-based active learning is that the former scans 
through the data sequentially and makes query decisions individually, 
whereas the latter evaluates and ranks the entire collection before selecting the best query.

Query Strategies
----------------

1) **Uncertainiy Sampling**: Learner will choose instances which it is least certain how to label.

3) **Query by Committee**: Using an ensemble of models to vote on which candidates to label.

Resources
----------
 * Active Learning Literature Survey. http://burrsettles.com/pub/settles.activelearning.pdf
 * Class Imbalance & Active Learning. https://pdfs.semanticscholar.org/7437/aae9bf347ab4ba4057f28df5f2eaf64d8fdc.pdf