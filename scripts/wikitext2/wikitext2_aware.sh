
echo "============= WikiText2, mode: aware; partition: shard-based noniid ============="
starttime=`date +'%Y-%m-%d %H:%M:%S'`

python main.py --mode aware \
              --dataset wikitext2 \
              --model_name transformer \
              --client_state real \
              --select_mode absmax \
              --rounds 1600 \
              --epoch 2 \
              --lr 0.01 \
              --frc 0.2 \
              --milestones 800 \
              --scheduler_name MultiStepLR \
              --project_name WikiText2 \
              --run_name=FedDSE-wikitext2 \
              --group_name noniid \
              --device 0

endtime=`date +'%Y-%m-%d %H:%M:%S'`     # Save the time in a human-readable format.
start_seconds=$(date --date="$starttime" +%s);
end_seconds=$(date --date="$endtime" +%s);      # Convert to a Unix timestamp to calculate script runtime.
echo 'running time is:' $((end_seconds-start_seconds)) 's'