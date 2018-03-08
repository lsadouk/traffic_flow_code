% ************************************************************************
% Plot the receptive fields (i.e., connection weights) of the second layer
%
% Alberto Testolin
% Computational Cognitive Neuroscience Lab
% University of Padova
% ************************************************************************

function [] = plot_L2(DN, n_hidden)

figure(2);
[v,h] = size(DN.L{1}.vishid);  % number of visible and hidden units
imgsize = sqrt(v);
if n_hidden > h
    n_hidden = h;
end
n_x = floor(sqrt(n_hidden)); n_y = n_x;
n_hidden = n_x * n_y;

for i_n = 1:n_hidden
    % Select (strong) inputs to L2(i_n)
    W2 = DN.L{2}.vishid(:,i_n);
    W2 = W2 .* (abs(W2) > 0.0);  % threshold
    % Select (strong) inputs to L1
    W1 = DN.L{1}.vishid;
    W1 = W1 .* (abs(W1) > 0.0);  % threshold
    
    ww = W1*W2;  % combine all L1 filters weighting them with L2
    ww = ww .* (abs(ww) > 0.0);  % threshold
    
    pl = subplot(n_y,n_x,i_n);
    position = get(pl, 'pos');
    position(3) = position(3) + 0.004;
    position(4) = position(4) + 0.004;
    set(pl, 'pos', position);
    imagesc(reshape(ww,imgsize,imgsize));
    colormap('gray'); axis square; axis off;
end
