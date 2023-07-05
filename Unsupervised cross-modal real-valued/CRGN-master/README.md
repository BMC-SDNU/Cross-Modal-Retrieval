# CRGN
Deep Relation Embedding for Cross-Modal Retrieval<br/>

This code is implemented by pytorch.<br/>
The meaning of file and dir:<br/>
final_CRGN and final_CRGN_f30k is related to MSCOCO and Flickr30K datasets, respectively.<br/>
runs:Save generated model and log <br/>
data.py:Process data<br/>
evaluation.py,evaluation_zyf.py:Evaluating model performance on Recall<br/>
model.py,model_zyf.py:Network structure<br/>
test.py:Test model <br/>
train.py:Train model <br/>
vocab.py:Generate vocabulary<br/>

How to train：python train.py --max_violation<br/>
How to test：python test.py<br/>
To reproduce the results in the paper, please follow the training procedure in the paper.
