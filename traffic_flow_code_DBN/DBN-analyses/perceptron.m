% ************************************************************************
% Simple perceptron implemented with the pseudo-inverse method
% NB: sometimes the backslash and the pinv give different results!
%
% Alberto Testolin
% Computational Cognitive Neuroscience Lab
% University of Padova
% ************************************************************************

function [weights, tr_accuracy, te_accuracy] = perceptron(tr_patterns, tr_labels, te_patterns, te_labels)

te_accuracy = 0;
tr_accuracy = 0;

% add biases
ONES = ones(size(tr_patterns, 1), 1);
tr_patterns = [tr_patterns ONES];

% train with pseudo-inverse
weights = tr_patterns\tr_labels;
%weights = (tr_labels'*pinv(tr_patterns'))';

% training accuracy
pred = tr_patterns*weights;
[~, max_act] = max(pred,[],2);
[r,~] = find(tr_labels'); % find which columns (rows in transpose) are 1
acc = (max_act == r);
tr_accuracy = mean(acc);

if ~isempty(te_patterns)
    % test accuracy
    ONES = ones(size(te_patterns, 1), 1);
    te_patterns = [te_patterns ONES];
    pred = te_patterns*weights;
    [~, max_act] = max(pred,[],2);
    [r,~] = find(te_labels');
    acc = (max_act == r);
    te_accuracy = mean(acc);
end

end
