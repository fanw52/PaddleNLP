model_item=gpt3
model=gpt
bs_item=16
fp_item=fp16
run_mode=DP
device_num=N4C32
max_epochs=200
num_workers=0

# get data
bash ./test_tipc/static/dp/${model}/benchmark_common/prepare.sh
# run
bash ./test_tipc/static/dp/${model}/benchmark_common/run_benchmark.sh ${model_item} ${bs_item} ${fp_item} ${run_mode} ${device_num} ${max_epochs} ${num_workers} 2>&1;
