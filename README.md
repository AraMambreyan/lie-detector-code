# Lie Detection 

This repository includes the code for a paper we submitted to ICPR 2022 which is a critique of 
the state-of-the-art papers in lie detection.

## Experiments
 
We train a machine learning classifier to predict sex labels.  On a test
data point, the classifier predicts sex and uses this as a
proxy for predicting deception – predicting lie if the predicted
sex is female and vice-versa. 

Using IDT features similarly to [[2]](#2), we achieve comparable results with this ad-hoc method. The notebook
for this experiment is `IDT.ipynb`. 

Using manually annotated micro-expressions, we achieve 65% using sex labels whereas training on lie/truth labels
78% is achieved which shows that even manually annotated micro-expressions are correlated with sex.
The experiment for this is in `micro-expressions.ipynb`.

The `helpers.py` contains useful functions for visualizing and running experiments. We needed fine-grained control
on the way cross-validation was run so we implemented  our own functions instead of using `sklearn`'s.


## Running

You need `numpy`, `sklearn`, `matplotlib` and `pandas` libraries
to run the code. Go to a directory where you 
want to clone the repository and run:

`git clone https://github.com/AraMambreyan/LieDetector-IDT.git`

Open `IDT.ipynb` or `micro-expressions.ipynb` notebooks. Change the `run_experiment_with_sex_labels` variable to the experiment
you'd like to run. Then click `run all` and our results will be reproduced.

## References
<a id="1">[1]</a> 
R. Mihalcea, V. P´erez-Rosas, M. Abouelenien and
M. Burzo. Deception detection using real-life trial data.
In *Proceedings of the 2015 ACM on International Conference on Multimodal Interaction*, page 59–66, 2015.

<a id="2">[2]</a> 
 Z. Wu, B. Singh, L. S. Davis, and V. S. Subrahmanian. Deception Detection in Videos.
In *AAAI*, 2018.
