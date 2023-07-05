echo "BERT"
echo "Flickr30K"
echo "evalaute f_model1"
# python evaluation_bert.py --model BERT/f_model1 --data_path "$DATA_PATH" --region_bbox_file "$REGION_BBOX_FILE" --feature_path "$FEATURE_PATH"
python evaluation_bert.py --model BERT/f_model1_ --data_path /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/data --region_bbox_file /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/flickr30k_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5 --feature_path /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/trainval/
mv sims_f.npy ./flickr_sims
mv sims_f_T.npy ./flickr_sims
echo "evalaute f_model2"
# python evaluation_bert.py --model BERT/f_model2 --data_path "$DATA_PATH" --region_bbox_file "$REGION_BBOX_FILE" --feature_path "$FEATURE_PATH"
python evaluation_bert.py --model BERT/f_model2_ --data_path /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/data --region_bbox_file /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/flickr30k_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5 --feature_path /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/trainval/
echo "ensemble and rerank!"
python rerank.py --data_name f30k
