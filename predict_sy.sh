# !/bin/bash
python predict.py --maxdisp 192 --with_spn --pretrained /home/lab3/work/lhx/code/AnyNet/checkpoint/kitti2015_ck/checkpoint.tar --left_images /home/lab3/work/lhx/data/kitti2015/testing/image_2/ --right_images /home/lab3/work/lhx/data/kitti2015/testing/image_3/ --save_dir /home/lab3/work/lhx/data/kitti2015/testing/result/ --datatype 2015 --evaluate --test_bsize 1
