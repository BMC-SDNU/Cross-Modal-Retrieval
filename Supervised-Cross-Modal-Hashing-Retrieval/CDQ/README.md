# aaai17-cdq

This is the Tensorflow (Version 0.11) implementation of AAAI-17 paper "Collective Deep Quantization for Efficient Cross-modal Retrieval". The descriptions of files in this directory are listed below:

- `cdq.py`: contains the main implementation of the proposed approach `cdq`.
- `train_script.py`: gives an example to show how to train `cdq` model. 
- `validation_script.py`: gives an example to show how to evaluate the trained quantization model.
- `run_cdq.sh`: gives an example to show the full procedure of training and evaluating the proposed approach `cdq`.

Data Preparation
---------------
In `data/nuswide/train.txt` and `data/nuswide/text_train.txt`, we give an example to show how to prepare image/text training data. In `data/nuswide/test.txt`, `data/nuswide/text_test.txt`, `data/nuswide/database.txt` and `data/nuswide/text_database.txt`, the list of testing and database images/texts could be processed during predicting procedure.

Training Model and Predicting
---------------
The [AlexNet](https://github.com/guerzh/tf_weights) is used as the pre-trained model. If the NUS\_WIDE dataset and pre-trained caffemodel is prepared, the example can be run with the following command:
```
"./run_cdq.sh"
```

Citation
---------------
    @inproceedings{DBLP:conf/aaai/CaoL0L17,
      author    = {Yue Cao and
                   Mingsheng Long and
                   Jianmin Wang and
                   Shichen Liu},
      title     = {Collective Deep Quantization for Efficient Cross-Modal Retrieval},
      booktitle = {Proceedings of the Thirty-First {AAAI} Conference on Artificial Intelligence,
                   February 4-9, 2017, San Francisco, California, {USA.}},
      pages     = {3974--3980},
      year      = {2017},
      crossref  = {DBLP:conf/aaai/2017},
      url       = {http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14499},
      timestamp = {Mon, 06 Mar 2017 11:36:24 +0100},
      biburl    = {http://dblp2.uni-trier.de/rec/bib/conf/aaai/CaoL0L17},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
