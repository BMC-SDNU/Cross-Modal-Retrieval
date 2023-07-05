
echo "GRU"
echo "Flickr30K"
echo "evalaute f_model1"
# python evaluation_bert.py --model GRU/f_model1 --data_path "$DATA_PATH" --region_bbox_file "$REGION_BBOX_FILE" --feature_path "$FEATURE_PATH"
python evaluation.py --model fmodel_1_ 
mv sims_f.npy ./flickr_sims
mv sims_f_T.npy ./flickr_sims
echo "evalaute f_model2"
# python evaluation_bert.py --model GRU/f_model2 --data_path "$DATA_PATH" --region_bbox_file "$REGION_BBOX_FILE" --feature_path "$FEATURE_PATH"
python evaluation.py --model fmodel_2_ 
echo "ensemble and rerank!"
python rerank.py --data_name f30k
