#!/bin/bash
source ~/.bashrc

ACCELERATE_CONFIG_PATHS=(amlt_configs/accelerate_config.yaml amlt_configs/accelerate_deepspeed_config.yaml)
if [[ -z "$WORLD_SIZE" ]]; then
    echo "WORLD_SIZE is not set, using 1"
    WORLD_SIZE=1
fi
if [[ -z "$NODE_RANK" ]]; then
    echo "NODE_RANK is not set, using 0"
    NODE_RANK=0
fi
NUM_GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
((NUM_TOTAL_GPUS = WORLD_SIZE * NUM_GPUS_PER_NODE))

echo "Setting up accelerate config:"
echo "ACCELERATE_CONFIG_PATHS: ${ACCELERATE_CONFIG_PATHS[@]}"
echo "NUM_TOTAL_GPUS: $NUM_TOTAL_GPUS"
echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

function modify_accelerate_config()
{
    local ACCELERATE_CONFIG_PATH=$1
    if [[ -z "$MASTER_ADDR" ]]; then
        echo "MASTER_ADDR is not set, using localhost"
        sed -i 's/main_process_ip.*//g' $ACCELERATE_CONFIG_PATH
        sed -i 's/main_process_port.*//g' $ACCELERATE_CONFIG_PATH
    else
        sed -i 's/main_process_ip.*/main_process_ip: '"$MASTER_ADDR"'/g' $ACCELERATE_CONFIG_PATH
        sed -i 's/main_process_port.*/main_process_port: '"$MASTER_PORT"'/g' $ACCELERATE_CONFIG_PATH
    fi

    sed -i 's/num_machines.*/num_machines: '"$WORLD_SIZE"'/g' $ACCELERATE_CONFIG_PATH
    sed -i 's/machine_rank.*/machine_rank: '"$NODE_RANK"'/g' $ACCELERATE_CONFIG_PATH

    sed -i 's/num_processes.*/num_processes: '"$NUM_TOTAL_GPUS"'/g' $ACCELERATE_CONFIG_PATH

    accelerate env --config_file $ACCELERATE_CONFIG_PATH
    # accelerate test --config_file $ACCELERATE_CONFIG_PATH  # It may cause bug..ValueError: To use a `DataLoader` in `split_batches` mode, the batch size (8) needs to be a round multiple of the number of processes (16).
}

for ACCELERATE_CONFIG_PATH in "${ACCELERATE_CONFIG_PATHS[@]}"; do
    if [[ -f "$ACCELERATE_CONFIG_PATH" ]]; then
        echo "ACCELERATE_CONFIG_PATH: $ACCELERATE_CONFIG_PATH exists, modifying it with env variables."
        modify_accelerate_config $ACCELERATE_CONFIG_PATH
    else
        echo "ACCELERATE_CONFIG_PATH: $ACCELERATE_CONFIG_PATH does not exist"
    fi
done