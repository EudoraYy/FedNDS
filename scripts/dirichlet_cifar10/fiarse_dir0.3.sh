echo "============= CIFAR10, mode: fiarse; partition: Dir0.3 ============="
starttime=`date +'%Y-%m-%d %H:%M:%S'`

python main.py --mode fiarse \
              --dataset cifar10 \
              --model_name masked_resnet18 \
              --partition_type dir \
              --par_alpha 0.3 \
              --client_state real \
              --rounds 1600 \
              --epoch 1 \
              --lr 0.01 \
              --weight_decay 0.0 \
              --momentum 0.0 \
              --project_name CIFAR10-Dir_0.3 \
              --run_name=FIARSE-cifar10-Dir0.3 \
              --group_name Dir_0.3 \
              --device 0

endtime=`date +'%Y-%m-%d %H:%M:%S'`     # Save the time in a human-readable format.
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);      # Convert to a Unix timestamp to calculate script runtime.
echo 'running time is:' $((end_seconds-start_seconds)) 's'