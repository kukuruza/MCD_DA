## For citycam adaptation.
```
CUDA_VISIBLE_DEVICES=1 && src_split='synthetic-Sept19' && tgt_split='real-Sept23-train' && basename='batch10' && \
  python adapt_trainer.py citycam citycam --src_split ${src_split} --tgt_split ${tgt_split} \
  --net drn_d_105 --batch_size 10 \
  --train_img_shape 64 64 --add_bg_loss --savename ${basename}

CUDA_VISIBLE_DEVICES=1 && src_split='synthetic-Sept19' && tgt_split='real-Sept23-train' && test_split='real-Sept23-test' && basename='batch10-onestep' && \
  for epoch in 1 2 3 4 5 6 7 8 9 10; do \
  python adapt_tester.py citycam train_output/citycam-${src_split}2citycam-${tgt_split}_3ch/pth/MCD-${basename}-drn_d_105-${epoch}.pth.tar \
  --net drn_d_105 --test_img_shape 64 64 --split ${test_split} --add_bg_loss; \
  done

src_split='synthetic-Sept19' && split='real-Sept23-train0.5' && test_split='real-Sept23-test' && subset='test' && basename='batch10' && \
  for epoch in 1 2 3 4 5 6 7 8 9 10; do \
  test_output_dir=citycam-${split}_only_3ch---citycam-${test_split}/${basename}-drn_d_105-${epoch}.tar && \
  ${HOME}/projects/shuffler/shuffler.py \
  -i /home/etoropov/src/MCD_DA/segmentation/test_output/${test_output_dir}/predictedtop.db --rootdir $CITY_PATH \
  evaluateSegmentationIoU \
  --gt_db_file $CITY_PATH/data/patches/Sept23-real/${subset}.db  \
  --gt_mapping_dict '{"<10": "background", ">245": "car"}'; \
  done
```

## For citycam source training on synthetic, testing on real.

```
CUDA_VISIBLE_DEVICES=1 && src_split='real-Sept23-train' && basename='batch10' && \
  python source_trainer.py citycam --split ${src_split} \
  --net drn_d_105 --batch_size 10 \
  --train_img_shape 64 64 --add_bg_loss --savename ${basename}

CUDA_VISIBLE_DEVICES=1 && split='real-Sept23-train' && test_split='real-Sept23-test' && basename='batch10-onestep' && \
  for epoch in 1 2 3 4 5 6 7 8 9 10; do \
  python source_tester.py citycam train_output/citycam-${split}_only_3ch/pth/${basename}-drn_d_105-${epoch}.pth.tar \
  --test_img_shape 64 64 --split ${test_split}; \
  done

# Evaluate
src_split='synthetic-Sept19' && tgt_split='real-Sept23-train' && test_split='real-Sept23-test' && subset='test' && basename='batch10-onestep' && \
  for epoch in 1 2 3 4 5 6 7 8 9 10; do \
  test_output_dir=citycam-${src_split}2citycam-${tgt_split}_3ch---citycam-${test_split}/MCD-${basename}-drn_d_105-${epoch}.tar && \
  ${HOME}/projects/shuffler/shuffler.py \
  -i /home/etoropov/src/MCD_DA/segmentation/test_output/${test_output_dir}/predictedtop.db --rootdir $CITY_PATH \
  evaluateSegmentationIoU \
  --gt_db_file $CITY_PATH/data/patches/Sept23-real/${subset}.db  \
  --gt_mapping_dict '{"<10": "background", ">245": "car"}'; \
  done
```

# Maximum Classifier Discrepancy for Domain Adaptation with Semantic Segmentation Implemented by PyTorch

<img src='../docs/result_seg.png' width=900/>  

***
## Installation
Use **Python 2.x**

First, you need to install PyTorch following [the official site instruction](http://pytorch.org/).

Next, please install the required libraries as follows;
```
pip install -r requirements.txt
```

## Usage
### Training
- Dataset
    - Source: GTA5 (gta), Target: Cityscapes (city)
- Network
    - Dilated Residual Network (drn_d_105)

We train the model following the assumptions above;
```
python adapt_trainer.py gta city --net drn_d_105
```
Trained models will be saved as "./train_output/gta-train2city-train_3ch/pth/normal-drn_d_105-res50-EPOCH.pth.tar"

### Test
```
python adapt_tester.py city ./train_output/gta-train2city-train_3ch/pth/normal-drn_d_105-res50-EPOCH.pth.tar
```

Results will be saved under "./test_output/gta-train2city-train_3ch---city-val/normal-drn_d_105-res50-EPOCH.tar"

<!-- 
#### CRF postprocessing
To use crf.py, you need to install pydensecrf. (https://github.com/lucasb-eyer/pydensecrf)

```
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

After you ran adapt_tester, you can apply crf as follows;

For validation data
```
python crf.py ./outputs/YOUR_MODEL_NAME/prob crf_output --outimg_shape 2048 1024
```

For test data
```
python crf.py ./outputs/YOUR_MODEL_NAME/prob crf_output --outimg_shape 1280 720
```

Optionally you can use raw img as follows;
```
python crf.py outputs/spatial-adapt-g-0.001000-7/prob  outputs/spatial-adapt-g-0.001000-7/label_crf_rawimg --raw_img_indir /data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_val_imgs
```

#### Visualize with Legend
After you ran adapt_tester, you can apply visualization_with_legend as follows;
```
python visualize_result.py --indir_list outputs/loss-weighted152-test-g-0.001000-k4-7/label/ outputs/psp04-test-g-0.001000-k4-9/label/ outputs/spatial-resnet101-testdata-g-0.001000-k4-11/label/ outputs/psp-test-g-0.001000-k4-28/label/ outputs/loss-weighted152-test-g-0.001000-k4-14/label --outdir merged
```
![](_static/vis_with_legend.png)

Results will be saved under "./outputs/YOUR_MODEL_NAME/vis_with_legend".
-->


### Evaluation
```
python eval.py city ./test_output/gta-train2city-train_3ch---city-val/normal-drn_d_105-res50-EPOCH.tar/label
```

## Reference codes
- https://github.com/Lextal/pspnet-pytorch
- https://github.com/fyu/drn
- https://github.com/meetshah1995/pytorch-semseg
- https://github.com/ycszen/pytorch-seg
