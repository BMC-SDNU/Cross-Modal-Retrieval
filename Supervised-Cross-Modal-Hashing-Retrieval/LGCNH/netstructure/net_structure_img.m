function net = net_structure_img(net,codelens)
    n = numel(net.layers) ;
    net.layers = net.layers(1:n-2);
    n = numel(net.layers) ;
    for i=1:n
        if ~isempty(net.layers{i}.weights)
            net.layers{i}.weights{1} = gpuArray(net.layers{i}.weights{1}) ;
            net.layers{i}.weights{2} = gpuArray(net.layers{i}.weights{2}) ;
        end
    end
    net.layers{i+1}.pad = [0,0,0,0];
    net.layers{i+1}.stride = [1,1];
    net.layers{i+1}.type = 'conv';
    net.layers{i+1}.name = 'fc8';
    net.layers{i+1}.weights{1} = gpuArray(0.01*randn(1,1,4096,codelens,'single'));
    net.layers{i+1}.weights{2} = gpuArray(0.01*randn(1,codelens,'single'));
    net.layers{i+1}.opts = {};
    net.layers{i+1}.dilate = 1;
end