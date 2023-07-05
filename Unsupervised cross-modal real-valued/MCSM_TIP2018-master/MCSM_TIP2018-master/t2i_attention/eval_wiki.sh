
# char cnn rnn
th ./t2i_attention/extract_wiki.lua \
  -data_dir ./t2i_attention/data \
  -num_caption 1 \
  -gpuid 3 \
  -ttype char \
  -model ./t2i_attention/trained_models/lm_sje_c10_hybrid_0.00010_1_1_trainids.txt_10000.t7 

