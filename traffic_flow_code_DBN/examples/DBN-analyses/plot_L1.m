% ************************************************************************
% Plot the receptive fields (i.e., connection weights) of the first layer
%
% Alberto Testolin
% Computational Cognitive Neuroscience Lab
% University of Padova
% ************************************************************************

function [] = plot_L1(DN, n_hidden)

figure(1);
[v,h] = size(DN.L{1}.vishid);  % number of visible and hidden units
imgsize = sqrt(v);
if n_hidden > h
    n_hidden = h;
end
n_x = floor(sqrt(n_hidden)); n_y = n_x;
n_hidden = n_x * n_y; % we plot receptive fields on a square grid
selected_idxs_hid = 1:n_hidden; % plot all selected hidden units

for i_n = 1:n_hidden
    % Select (strong) inputs to L1(i_n)
    hidden_idx = selected_idxs_hid(i_n);
    W1 = DN.L{1}.vishid(:,hidden_idx);
    W1 = W1 .* (abs(W1) > 0.0);  % we can define a threshold
    
    pl = subplot(n_y,n_x,i_n);
    position = get(pl, 'pos');
    position(3) = position(3) + 0.004;
    position(4) = position(4) + 0.004;
    set(pl, 'pos', position);
    imagesc(reshape(W1,imgsize,imgsize));
    colormap('gray'); axis square; axis off;
end

