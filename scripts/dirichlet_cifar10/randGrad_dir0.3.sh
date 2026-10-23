echo "============= CIFAR10, mode: randGrad; partition: Dir0.3 ============="
starttime=`date +'%Y-%m-%d %H:%M:%S'`

python main.py --mode randGrad \
              --dataset cifar10 \
              --model_name resnet18 \
              --partition_type dir \
              --par_alpha 0.3 \
              --client_state real \
              --select_mode l2 \
              --rounds 1600 \
              --epoch 1 \
              --lr 0.01 \
              --milestones 800 \
              --scheduler_name MultiStepLR \
              --project_name CIFAR10-Dir_0.3 \
              --run_name=FedNDS-cifar10-Dir0.3 \
              --group_name Dir_0.3 \
              --device 0

endtime=`date +'%Y-%m-%d %H:%M:%S'`     # Save the time in a human-readable format.
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);      # Convert to a Unix timestamp to calculate script runtime.
echo 'running time is:' $((end_seconds-start_seconds)) 's'