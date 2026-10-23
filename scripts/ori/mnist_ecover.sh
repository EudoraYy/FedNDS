for spu in 4 2
  do
    echo "============= MNIST, mode: ecover; partition: L_"$spu " ============="
    starttime=`date +'%Y-%m-%d %H:%M:%S'`
    python main.py --mode ecover \
                  --dataset mnist \
                  --model_name conv \
                  --shardperuser $spu \
                  --client_state real \
                  --rounds 1600 \
                  --epoch 1 \
                  --lr 0.01 \
                  --milestones 800 \
                  --scheduler_name MultiStepLR \
                  --project_name FedNDS-compare \
                  --run_name=FedEcover-mnist-L$spu \
                  --group_name L_$spu \
                  --device 0 \

    endtime=`date +'%Y-%m-%d %H:%M:%S'`     # Save the time in a human-readable format.
    start_seconds=$(date --date="$starttime" +%s);
    end_seconds=$(date --date="$endtime" +%s);      # Convert to a Unix timestamp to calculate script runtime.
    echo 'running time is:' $((end_seconds-start_seconds)) 's'
  done