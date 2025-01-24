# Neural Network and Deep Learning 2024
## Giorgio Longobardo e Claudio Riccio

Traccia scelta: 2:

Consider the raw images from the MNIST dataset as input. This is a classification problem 
with C classes, where C= 10. Extract a global dataset of N pairs, and divide it appropriately 
into training and test sets (consider at least 10,000 elements for the training set and 2,500 for 
the test set). Use resilient backpropagation (RProp) as the weight update algorithm (batch 
update). Study the learning process of a neural network (e.g., epochs required for learning, 
error trend on training and validation sets, accuracy on the test set) with a single layer of 
internal nodes, varying  the number  of internal nodes  (selecting  at least five  different 
dimensions)   and   using   cross-entropy   loss   with   soft-max.   Select   and   keep   all   other 
hyperparameters constant, such as activation functions and RProp parameters. If necessary, 
due to computational time and memory constraints, you can reduce the dimensions of the 
raw MNIST dataset images (e.g., using the imresize function in MATLAB).

Traccia 6:

Consider the raw images from the MNIST dataset as input. This is a classification problem
with C classes, where C= 10. Extract a global dataset of N pairs, and divide It appropriately
into training and test sets (consider at least 10,000 elements for the training set and 2,500 for
the test set). Following the article “Empirical evaluation of the improved RProp learning
algorithms, Christian Igel, Michael Husken, Neurocomputing, 2003”, compare the classic
resilient backpropagation (RProp) with at least two proposed variants of the algorithm as
weight update methods (batch update). Fix the activation function and the number of
internal nodes (at least three different dimensions), and compare the results obtained with
the different learning algorithms. If necessary, due to computational time and memory
constraints, you can reduce the dimensions of the raw MNIST dataset images (e.g., using the
imresize function in MATLAB).

Riferimento a:

https://christian-igel.github.io/paper/EEotIRLA.pdf

(dal pdf) \
Da “Neural Networks for Pattern Recognition”
> Section 1: 1.1,1.2, 1.3, 1.4., 1.8,1.9 \
> Section 3: 3.1, 3.2, 3.5 \
> Section 4: 4.1, 4.2, 4.3, 4.8 \
> Section 6: 6.1,6.6, 6.7, 6.9 \
> Section 7: 7.1, 7.4, 7.5 \
> Section 8: 8.1, 8.2,8.5 \

Da “Deep Learning: Foundations and concepts”
> Section 6: 6.2, 6.2.1, 6.2.2, 6.2.3, 6.3, 6.3.1, 6.3.2, 6.3.3 \
> Section 10: 10.1, 10.2 \
> Section 12: 12.1


BACK PROPAGO
OGAPORP KCAB
