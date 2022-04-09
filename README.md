# Lie Detection 

This repository includes the code for a critique of the state-of-the-art papers in lie detection.

## Experiments
 
We train a machine learning classifier to predict sex labels.  On a test
data point, the classifier predicts sex and uses this as a
proxy for predicting deception – predicting lie if the predicted
sex is female and vice-versa. This ad-hoc classifier simulates a classifier which uses nothing but dataset bias.

For the Real-life Trial dataset [[1]](#1), using IDT features similarly to [[2]](#2), 
we achieve comparable results with this ad-hoc method. The notebook
for this experiment is `IDT.ipynb`. Using manually annotated micro-expressions, 
we achieve 65% using sex labels whereas training on lie/truth labels we achieve
78% which shows that even manually annotated micro-expressions are correlated with sex.
The experiment for this is in `micro-expressions.ipynb`.

For the Bag-of-Lies dataset, using gaze features, we achieve 54.4% using the ad-hoc which is statistically significant.
Using features such as audio and video should give considerably higher accuracies. The experiment for this is in 
`Bag-of_Lies.ipynb`. To run this experiment, you need to sign a [license agreement to obtain the data](http://iab-rubric.org/index.php/bag-of-lies)
and then put the data under the `BagOfLies` directory.

## Reproducing

You need `numpy-1.18.1`, `scikit-learn-0.22.1`, `matplotlib-3.1.3` and `pandas-1.0.1` libraries
to run the code (I only used standard functions so different versions will likely work). Go to a directory where you 
want to clone the repository and run:

`git clone https://github.com/AraMambreyan/LieDetector-IDT.git`

Open the notebook for the experiment you want to run. Change the `run_experiment_with_sex_labels` variable to the experiment
you'd like to run. Then click `run all` and our results will be reproduced.

## References
<a id="1">[1]</a> 
R. Mihalcea, V. P´erez-Rosas, M. Abouelenien and
M. Burzo. Deception detection using real-life trial data.
In *Proceedings of the 2015 ACM on International Conference on Multimodal Interaction*, page 59–66, 2015.

<a id="2">[2]</a> 
 Z. Wu, B. Singh, L. S. Davis, and V. S. Subrahmanian. Deception Detection in Videos.
In *AAAI*, 2018.

<a id="3">[3]</a> 
 V. Gupta, M. Agarwal, M. Arora, T. Chakraborty, R. Singh, and
M. Vatsa, “Bag-of-lies: A multimodal dataset for deception detection,”
in IEEE Conference on Computer Vision and Pattern Recognition
Workshops (CVPR), 2019, pp. 83–90
