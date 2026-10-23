for spu in 20 40
do
  echo "============= TinyImagenet, mode: fiarse; partition: L_"$spu " ============="
  starttime=`date +'%Y-%m-%d %H:%M:%S'`

  python main.py --mode fiarse \
                --dataset tinyimagenet \
                --model_name masked_resnet34 \
                --partition_type patho \
                --shardperuser $spu \
                --client_state real \
                --rounds 1600 \
                --epoch 1 \
                --lr 0.01 \
                --weight_decay 0.0 \
                --momentum 0.0 \
                --project_name FedNDS-compare \
                --run_name=FIARSE-tiny-L_$spu \
                --group_name L_$spu \
                --device 0

  endtime=`date +'%Y-%m-%d %H:%M:%S'`     # Save the time in a human-readable format.
  start_seconds=$(date --date="$starttime" +%s);
  end_seconds=$(date --date="$endtime" +%s);      # Convert to a Unix timestamp to calculate script runtime.
  echo 'running time is:' $((end_seconds-start_seconds)) 's'
done