model_path=llava-hf/llava-1.5-7b-hf
model_name=llava1.5-7b

# # dataset_root=/mnt/bn/lyl-test/mlx/users/luoyulin/playground/code/ssdllm/ICTC/dataset/stl10
# # dataset_name=stl10_0.1
# dataset_root=/mnt/bn/lyl-test/mlx/users/luoyulin/playground/code/ssdllm/ICTC/dataset
# dataset_name=imagenet
mkdir -p ${dataset_root}/${dataset_name}/split/${model_name}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python infer_batch.py \
        --model-name ${model_name} --model-path ${model_path} --dataset-root ${dataset_root} --dataset-name ${dataset_name} \
        --query "USER: <image>\nPlease describe the image in detail with the keyword {class_name} ASSISTANT:" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX & \
done