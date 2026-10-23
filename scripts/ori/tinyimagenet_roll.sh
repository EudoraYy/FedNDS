for spu in 20 40
  do
    python main.py --mode roll \
                --dataset tinyimagenet \
                --model_name resnet34 \
                --shardperuser $spu \
                --client_state real \
                --rounds 1600 \
                --epoch 1 \
                --lr 0.01 \
                --milestones 800 \
                --scheduler_name MultiStepLR \
                --project_name FedNDS-compare \
                --run_name FedRolex-tinyimagenet-L$spu \
                --group_name L_$spu \
                --device 0
  done