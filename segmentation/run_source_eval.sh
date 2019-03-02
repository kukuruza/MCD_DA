#split='synthetic-Jan28-train'
#basename='normal'
test_split='real-Sept23-test'
gt_path=$CITY_PATH/data/patches/Sept23-real/test-pic.db


test_dir=${HOME}/src/MCD_DA/segmentation/test_output/citycam-${split}_only_3ch---citycam-${test_split}

for f in $(ls -d ${test_dir}/${basename}-drn_d_105-*.tar); do
  echo 'Working on directory'$f

  # Extract the epoch from the filename.
  epoch=$(echo $f | sed -n -e "s/.*105-//p" | sed -ne "s/\.tar//p")
  echo 'Epoch='$epoch

  test_epoch_dir=${test_dir}/${basename}-drn_d_105-${epoch}.tar

  # Evaluate.
  ${HOME}/projects/shuffler/shuffler.py \
    -i ${test_epoch_dir}/predictedtop.db \
    -o ${test_epoch_dir}/evaluated.db \
    --rootdir $(dirname $gt_path) \
    evaluateSegmentationIoU \
    --gt_db_file $gt_path \
    --gt_mapping_dict '{0: "background", 255: "car"}' \
    --out_dir ${test_dir}/${basename} \
    --out_summary_file summary.txt \
    --out_prefix $epoch \
    \| \
    plotHistogram --sql "SELECT score FROM images" --xlabel IoU --bins 50 --ylog --xlim 0.5 1.0 \
    --out_path ${test_epoch_dir}/iou_hist.png

done

ffmpeg -y -f image2  -framerate 1  -pattern_type glob \
  -i ${test_dir}/${basename}-drn_d_105-"*".tar/iou_hist.png \
  -pix_fmt rgb24  ${test_dir}/${basename}/iou_hist.gif
