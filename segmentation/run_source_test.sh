#split='synthetic-Jan28-train'
#basename='normal'
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

