CKPT_DIR=/home/qiuku/qk_data/svox2/out/result_mv_nosparse
mkdir -p $CKPT_DIR

python opt.py \
/home/qiuku/qk_data/svox2/data/scannet_0000_mv \
--train_dir $CKPT_DIR \
--config /home/qiuku/svox2/opt/configs/sc_nosparse.json