GPU_IDS=$1

DATAROOT=./datasets
NAME=WDcascadeNet
MODEL=WDcascadeNet
DATASET_MODE=deepcrack

BATCH_SIZE=1
NORM=batch
LOAD_WIDTH=640
LOAD_HEIGHT=360
NUM_CLASSES=1
NUM_TEST=10000

DATASET=CRACK500

python3 test.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --norm ${NORM} \
  --num_test ${NUM_TEST}\
  --display_sides 1\
  --dataset ${DATASET}\
  --img_dim ${img_dim}\
  --load_width ${LOAD_WIDTH} \
  --load_height ${LOAD_HEIGHT} 
