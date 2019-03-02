#split='synthetic-Jan28-train'
#basename='normal'
#epochs=10000000

python source_trainer.py citycam --split ${split} --net drn_d_105 --batch_size 10 \
  --train_img_shape 64 64 --add_bg_loss --savename ${basename}  --weight_yaw 0 --freq_log 50 \
  --epochs $epochs --freq_checkpoint 0.1

test_split='real-Sept23-test'

train_dir=${HOME}/src/MCD_DA/segmentation/train_output/citycam-${split}_only_3ch/pth

for f in $(ls -d ${train_dir}/${basename}-drn_d_105-*.tar); do
  echo 'Working on directory'$f

  # Extract the epoch from the filename.
  epoch=$(echo $f | sed -n -e "s/.*105-//p" | sed -ne "s/\.pth\.tar//p")
  echo 'Epoch='$epoch

  # Test.
  python source_tester.py citycam \
    train_output/citycam-${split}_only_3ch/pth/${basename}-drn_d_105-${epoch}.pth.tar \
    --test_img_shape 64 64  --split ${test_split}

done

