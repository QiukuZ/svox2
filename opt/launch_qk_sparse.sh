CKPT_DIR=/home/qiuku/ssd_data/plenoxel_out/sparse_si_test
mkdir -p $CKPT_DIR

python opt.py \
--data_dir /home/qiuku/ssd_data/scannet_0000_all \
--train_dir $CKPT_DIR \
--config /home/qiuku/code/svox2/opt/configs/sc_sparse.json