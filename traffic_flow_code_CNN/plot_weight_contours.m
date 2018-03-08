load('./result_data/data_D07_US101N20d_20by10_minP_R_1chan_WeigLossNo_convs_5pts/net-epoch-15.mat')
weights = net.layers{1,1}.weights{1,1}; % size:  3     3     1    32

nb_weights = size(weights,4);

for i=1:nb_weights  % 1:nb_weights
    figure,
    contourf(weights(:,:,:,i)),
    title(strcat('contour of weight N',int2str(i)));
end