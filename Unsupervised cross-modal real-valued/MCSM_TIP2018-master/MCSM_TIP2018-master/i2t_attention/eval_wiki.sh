
# char cnn rnn
th ./i2t_attention/extract_wiki.lua \
  -data_dir ./i2t_attention/data \
  -num_caption 1 \
  -ttype char \
  -gpuid 1 \
  -model ./i2t_attention/trained_models/lm_sje_c10_hybrid_0.00010_1_1_trainids.txt_10000.t7

