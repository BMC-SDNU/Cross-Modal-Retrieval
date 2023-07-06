function net = gcn_net(dy,codelens)
    net.layers{1}.weights{1} = gpuArray(0.01*randn(1,1,dy,codelens,'single'));
    net.layers{1}.weights{2} = gpuArray(0.01*randn(1,codelens,'single'));
    net.layers{1}.pad = [0,0,0,0];
    net.layers{1}.stride = [1,1];
    net.layers{1}.type = 'conv';
    net.layers{1}.name = 'fc1';
    net.layers{1}.opts = {};
    net.layers{1}.dilate = 1;
    
    net.layers{2}.type = 'relu';
    net.layers{2}.weights = {};
    net.layers{2}.name = 'relu1';
    net.layers{2}.leak = 0;
    net.layers{2}.precision = false;
    
    net.layers{3}.pad = [0,0,0,0];
    net.layers{3}.stride = [1,1];
    net.layers{3}.type = 'conv';
    net.layers{3}.name = 'fc2';
    net.layers{3}.weights{1} = gpuArray(0.01*randn(1,1,codelens,codelens,'single'));
    net.layers{3}.weights{2} = gpuArray(0.01*randn(1,codelens,'single'));
    net.layers{3}.opts = {};    
    net.layers{3}.dilate = 1;
    
end