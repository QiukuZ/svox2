CKPT_DIR=/home/qiuku/qk_data/svox2/out/result_all_zeros
mkdir -p $CKPT_DIR

python opt.py \
/home/qiuku/qk_data/svox2/data/scannet_0000_all \
--train_dir $CKPT_DIR \
--config /home/qiuku/svox2/opt/configs/sc_zeros.json