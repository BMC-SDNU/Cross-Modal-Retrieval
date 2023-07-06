[n, ~] = size(dataset.YDatabase);
anchor_image = dataset.XDatabase(randsample(n, n_anchors),:); 
anchor_text = dataset.YDatabase(randsample(n, n_anchors),:);
dataset.XDatabase = RBF_fast(dataset.XDatabase',anchor_image'); dataset.XTest = RBF_fast(dataset.XTest',anchor_image'); 
dataset.YDatabase = RBF_fast(dataset.YDatabase',anchor_text');  dataset.YTest = RBF_fast(dataset.YTest',anchor_text'); 