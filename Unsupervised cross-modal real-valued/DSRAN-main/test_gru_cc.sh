echo "GRU"
echo "MSCOCO"
echo "evalaute cc_model1"
# python evaluation_bert.py --model GRU/cc_model1 --fold --data_path "$DATA_PATH" --region_bbox_file "$REGION_BBOX_FILE" --feature_path "$FEATURE_PATH"
python evaluation.py --model GRU/cc_model1 --fold --data_path /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/data --region_bbox_file /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/joint-pretrain/COCO/region_feat_gvd_wo_bgd/coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5 --feature_path /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/joint-pretrain/COCO/region_feat_gvd_wo_bgd/feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval
mv sims_full_0.npy sims_full_1.npy sims_full_2.npy sims_full_3.npy sims_full_4.npy sims_full_5k.npy ./coco_sims
echo "evalaute cc_model2"
# python evaluation_bert.py --model GRU/cc_model2 --fold --data_path "$DATA_PATH" --region_bbox_file "$REGION_BBOX_FILE" --feature_path "$FEATURE_PATH"
python evaluation.py --model GRU/cc_model2 --fold --data_path /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/data --region_bbox_file /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/joint-pretrain/COCO/region_feat_gvd_wo_bgd/coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5 --feature_path /remote-home/lyli/Workspace/meltdown/burneddown/ECCV/joint-pretrain/COCO/region_feat_gvd_wo_bgd/feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval
echo "ensemble and rerank!"
echo "fold5-1K"
python rerank.py --data_name coco --fold
echo "5K"
python rerank.py --data_name coco
