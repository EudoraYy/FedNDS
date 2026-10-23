for spu in 10 20
do
    echo "============= CIFAR100, mode: hetero; partition: L_"$spu " ============="
    starttime=`date +'%Y-%m-%d %H:%M:%S'`

    python main.py --mode hetero \
                  --dataset cifar100 \
                  --model_name resnet18 \
                  --shardperuser $spu \
                  --client_state real \
                  --rounds 1600 \
                  --epoch 1 \
                  --lr 0.01 \
                  --milestones 800 \
                  --scheduler_name MultiStepLR \
                  --project_name FedNDS-compare \
                  --run_name=HeteroFL-cifar100-L$spu \
                  --group_name L_$spu \
                  --device 0

    endtime=`date +'%Y-%m-%d %H:%M:%S'`     # Save the time in a human-readable format.
    start_seconds=$(date --date="$starttime" +%s);
    end_seconds=$(date --date="$endtime" +%s);      # Convert to a Unix timestamp to calculate script runtime.
    echo 'running time is:' $((end_seconds-start_seconds)) 's'
done