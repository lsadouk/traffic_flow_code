weights = net.nets{3}.W;

[~, order] = sort(sum(weights .^2), 'descend');
colormap gray
for i = 1:20
    subplot(5, 4, i);
    imagesc(reshape(weights(:, order(i)), 25 , 20)); %imagesc(reshape(weights(:, order(i)), 28, 28));
    axis image
    axis off
end