CUDA_VISIBLE_DEVICES=3 python train.py > logs/n1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python tools/train.py -c configs/rtdetr/rtdetr_r50vd_ssdd.yml -t output/ckpts/rtdetr_r50vd_6x_coco_from_paddle.pth > logs/new_6x_0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python tools/train.py -c configs/rtdetr/rtdetr_r50vd_ssdd.yml -t output/ckpts/rtdetr_r50vd_6x_coco_from_paddle.pth > logs/new_6x_1.log 2>&1 &
# test only
# CUDA_VISIBLE_DEVICES=3 python tools/train.py -c configs/rtdetr/rtdetr_r50vd_ssdd.yml -t output/new_1x_auxhead_solo/checkpoint0012.pth --test-only