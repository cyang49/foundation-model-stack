BATCH_SIZE=${BATCH_SIZE:-128}
INPUT_LEN=128
OUTPUT_LEN=128
MODEL_SIZE=7b
OUT_DIR=./output


BASE_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=1 inference-new.py --distributed --batch_size ${BATCH_SIZE} --max_new_tokens ${OUTPUT_LEN} --max_seq_len ${INPUT_LEN} --architecture="llama" --variant=${MODEL_SIZE} --model_source="hf" --tokenizer=/net/storage149/autofs/css22/zhuoran/data/llama-70b-tokenizer/tokenizer.model --model_path=/net/storage149/autofs/css22/nmg/models/llama-${MODEL_SIZE}/fms/*.pth"

COMPILE_CUDAGRAPH="--compile --compile_mode=reduce-overhead"
COMPILE_INDUCTOR="--compile --compile_mode=max-autotune"

mkdir -p ${OUT_DIR}

PREC=fp8
for MODE in eager cudagraph inductor; do
for FP8_LINEAR_TYPE in dasw sw; do

    FP8_OPTS="--fp8 --fp8_linear_type=${FP8_LINEAR_TYPE}"

    if [[ "$MODE" = "eager" ]]; then
        COMPILE_OPTS=
    elif [[ "$MODE" = "cudagraph" ]]; then
        COMPILE_OPTS=${COMPILE_CUDAGRAPH}
    elif [[ "$MODE" = "inductor" ]]; then
        COMPILE_OPTS=${COMPILE_INDUCTOR}
    else
        echo "Unsupported MODE=$MODE"
        exit -1
    fi
    echo "${BASE_CMD} ${COMPILE_OPTS} ${FP8_OPTS}" 
    eval ${BASE_CMD} ${COMPILE_OPTS} ${FP8_OPTS}\
    	|& tee ${OUT_DIR}/out_fms_llama-${MODEL_SIZE}_bs${BATCH_SIZE}_ii${INPUT_LEN}_oo${OUTPUT_LEN}_${PREC}_${FP8_LINEAR_TYPE}_${MODE}.txt

done
done

PREC=fp16
for MODE in eager cudagraph inductor; do

    if [[ "$MODE" = "eager" ]]; then
        COMPILE_OPTS=
    elif [[ "$MODE" = "cudagraph" ]]; then
        COMPILE_OPTS=${COMPILE_CUDAGRAPH}
    elif [[ "$MODE" = "inductor" ]]; then
        COMPILE_OPTS=${COMPILE_INDUCTOR}
    else
        echo "Unsupported MODE=$MODE"
        exit -1
    fi
    echo "${BASE_CMD} ${COMPILE_OPTS}"
    eval ${BASE_CMD} ${COMPILE_OPTS} \
    	|& tee ${OUT_DIR}/out_fms_llama-${MODEL_SIZE}_bs${BATCH_SIZE}_ii${INPUT_LEN}_oo${OUTPUT_LEN}_${PREC}_${MODE}.txt
    
done
