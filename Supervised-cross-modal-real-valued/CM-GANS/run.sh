echo 'TRAINING START...........'
sh scripts/train_pascal.sh
echo 'EXTRACTION START...........'
sh scripts/extract_pascal.sh
echo 'EVALUATION START...........'
matlab -nodesktop -nosplash -r "run eval/evalMAP;run eval/evalAllMAP;quit;"