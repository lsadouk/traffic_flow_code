%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
If you find this code useful, please cite our work as:

Zorzi, M., Testolin, A. and Stoianov, I. (2013)
'Modeling language and cognition with deep unsupervised learning:
a tutorial overview.' Frontiers in Psychology.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

These scripts perform some useful analyses in order to evaluate the
internal representations learned by a hierarchical generative model.


Plot receptive fields at different levels of the hierarchy, given as
input a deep belief network and the desired number of hidden units:

plot_L1(DN, n_hidden)
plot_L2(DN, n_hidden)
plot_L3(DN, n_hidden)


Perform a read-out using a simple linear classifier, given as input a
set of training and test patterns with corresponding labels. The function
gives as output the weights of the classifier and the training and test
accuracies:

[W, tr_acc, te_acc] = perceptron(tr_patt, tr_labels, te_patt, te_labels)
