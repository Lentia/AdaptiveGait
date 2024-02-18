# # **************** For CASIA-B ****************
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs ./configs/adaptivegait/adaptivegait.yaml --phase test


# # **************** For OUMVLP ****************
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 opengait/main.py --cfgs ./configs/adaptivegait/adaptivegait_OUMVLP.yaml --phase test