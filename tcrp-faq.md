# TCRP Frequently Asked Questions (FAQ) List

The following is a list of commonly asked questions, along with brief answers. This is a living document, which is updated continually as we and others research the application of few-shot learning to biomedical problems. 

## Learning problem

### What is meta-learning?

Meta-learning is a branch of machine learning that aims to develop algorithms that learn the general ability to learn and adapt when faced with a new problem.

### What is few-shot learning?

The premise of "few-shot" learning is to build a model that can quickly adapt from a domain with plentiful data to a domain with few data. 

## What is MAML? 

MAML, or Model Agnostic Meta-Learning, is a few-shot learning model that aims to train a base model that can rapidly adapt to few, new examples. MAML achieves this by training on a variety of learning tasks. Instead of updating parameters on the loss based on the model accuracy of a single task, MAML updates the model based on its ability to quickly adapt to the set of learning tasks.

### How was few-shot learning applied to precision medicine in our study?

We evaluated few-shot learning for prediction of growth response of a tumor cell line based on its genetic mutation profile and gene expression profile.  Each cell line was exposed to many different drugs or gene knockouts by CRISPR. Given response data for tumor cells from many different tissues of origin, the main question of our study was whether a predictive model pre-trained in some tissue types was able to transfer to make predictions in a new tissue type it had not yet seen. Apart from how well the model performed initially, we were interested in how quickly it could learn to make good predictions as it was shown a small number k of responses from the new tissue type. 

### How do we define a single “learning task”?

We defined a single learning task as a single withheld tissue (t) with k examples provided. The model was first trained using data from all other tissue types (pretraining phase). Then, the model was fine-tuned (few-shot learning phase) given only k samples from the new tissue (t). In other words, a single withheld tissue (t) with k examples constitute a single task that is independent of another withheld tissue or a different choice of k. As such, hyperparameter selection for each of these models are independent of one another as well.

Why is a single drug or single CRISPR knockout not a learning task?

A single drug or CRISPR knockout is not a single learning task because we need to withhold all but k samples of the withheld categories. If we did so for all tissues, we would not have any data remaining to pretrain the base model.

### Why is each k a different learning task?

Each k represents a different scenario for the learning algorithm. We are constructing a scenario where the algorithm has access to 1 sample (k = 1) and another where the algorithm has access to 2 samples (k = 2). We do not require the algorithm to learn from each of these few-shot samples sequentially. In other words, the model for k = 2 does not depend on k = 1.

The basic idea behind MAML is to pretrain a single base-model that can learn rapidly (in a single gradient descent step in this work) on the few-shot examples. As a result, we are evaluating the model on its ability to learn (again in a single gradient descent step) on the few-shot samples.

## Model evaluation

### How is the test set constructed?

The test set consists of all cell lines from tissue (t) that are not withheld. For the original paper, the test set is different for each k, such that the test set for the k = 10 learning task has 9 fewer cell lines than for the k = 1 task, since those 9 cell lines are used for training. However, in this reproduction repository, we have kept all model evaluations consistent by always restricting the test set to the number of cell lines that remain for a tissue type after withholding 10 cell lines. To robustly estimate model performance, we repeated the test set evaluation 20 times for each learning task.

### How are hyperparameters selected?

We selected hyperparameters based on the performance on the few-shot learning training set for each learning task. This means the model selected for k = 2 may be different than for other k (e.g. k = 10) because they are two different learning tasks. If instead we required each to model from k=0 to k=10 to have the same hyperparameter, we would either have significant data leakage or be ignoring test data. For instance, if we selected samples based on k=10, then all other learning tasks would gain access to test data they should not have seen. Conversely, if we selected samples based on k=0, then we would be discarding valuable test data.

## Miscellaneous

### What is the “tcrp-reproduce” repository and how does it relate to the paper?

This repository is not the repository used to produce the original published paper. This code base is available to provide instructions to construct the data, train few-shot learning models, and select models.
