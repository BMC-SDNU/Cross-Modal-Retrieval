#!/usr/bin/env bash
echo 'demo code for wiki dataset'
echo 'training i2t model...'
sh ./i2t_attention/train_wiki.sh
echo 'i2t model training done!'
echo 'extracting feature using i2t model...'
sh ./i2t_attention/eval_wiki.sh
echo '[i2t] feature extraction done!'
echo 'training t2i model...'
sh ./t2i_attention/train_wiki.sh
echo 't2i model training done!'
echo 'extracting feature using t2i model...'
sh ./t2i_attention/eval_wiki.sh
echo '[t2i] feature extraction done!'
