CKPT_DIR=/home/qiuku/ssd_data/plenoxel_out/debug_crop_test
mkdir -p $CKPT_DIR

python opt/pipeline.py \
--data_dir /home/qiuku/ssd_data/scannet_0000_all \
--train_dir $CKPT_DIR \
--config /home/qiuku/code/svox2/opt/configs/sc_sparse_crop.json \
--config_convonet /home/qiuku/code/svox2/convonet/configs/scannet_0000_all_crop.yaml