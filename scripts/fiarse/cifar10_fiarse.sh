for spu in 4 2
do
  echo "============= CIFAR10, mode: fiarse; partition: L_"$spu " ============="
  starttime=`date +'%Y-%m-%d %H:%M:%S'`
  # Consistent with the original paper, we set the weight_decay parameter to 0.0.
  # If weight_decay is not 0.0, FIARSE would no longer involve truly sparse uploads,
  # which would make comparisons unfair.
  python main.py --mode fiarse \
                --dataset cifar10 \
                --model_name masked_resnet18 \
                --partition_type patho \
                --shardperuser $spu \
                --client_state real \
                --rounds 1600 \
                --epoch 1 \
                --lr 0.01 \
                --weight_decay 0.0 \
                --momentum 0.0 \
                --project_name FedNDS-compare \
                --run_name=FIARSE-cifar10-L_$spu \
                --group_name L_$spu \
                --device 0

  endtime=`date +'%Y-%m-%d %H:%M:%S'`
  start_seconds=$(date --date="$starttime" +%s);
  end_seconds=$(date --date="$endtime" +%s);
  echo 'running time is:' $((end_seconds-start_seconds)) 's'
done
