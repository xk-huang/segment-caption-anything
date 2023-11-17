#!/bin/bash

shopt -s expand_aliases

alias breakpoint='
    while read -p"Debugging(Ctrl-d to exit)> " debugging_line
    do
        eval "$debugging_line"
    done'

export TOKENIZERS_PARALLELISM=false
export COLUMNS_FOR_VDTK=4096  # we need this to avoid truncated output in vdtk score, use by `COLUMNS=$COLUMNS_FOR_VDTK vdtk ...`

RED='\033[0;31m'
GREEN='\033[0;32m'
LIGHT_GREEN='\033[1;32m'
NC='\033[0m' # No Color

# Prepare and check parameters
IMAGE_B64_TSV_PATH=/path/to/image_b64.tsv
MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT=scripts/tools/merge_img_tsv_into_json_for_vdtk.py
POST_PROCESS_MULTI_CANDIDATES_SCRIPT=scripts/tools/post_process_multi_candidates_for_vdtk.py

echo -e "${LIGHT_GREEN}Env args${NC}:"
echo -e "\t${GREEN}DRY_RUN${NC}: ${DRY_RUN}"
echo -e "\t${GREEN}ONLY_GATHER${NC}: ${ONLY_GATHER}"
echo -e "\t${GREEN}ONLY_EVAL${NC}: ${ONLY_EVAL}"
echo -e "\t${GREEN}SKIP_CLIP_RECALL${NC}: ${SKIP_CLIP_RECALL}"
echo -e "\t${GREEN}DEBUG${NC}: ${DEBUG}"
echo -e "\t${GREEN}NO_POST_PROCESS${NC}: ${NO_POST_PROCESS}"

# NOTE: shell compare
# -n: string is not null.
# -z: string is null, that is, has zero length
if [[ -n $ONLY_GATHER ]] && [[ -n $ONLY_EVAL ]]; then
    echo -e "${RED}Error: ${NC}ONLY_GATHER and ONLY_EVAL cannot be on at the same time."
    exit 1
elif [[ -n $ONLY_GATHER ]] && [[ -z $1 ]]; then
    echo -en "Usage: ONLY_GATHER=1 ./eval_suite.sh ${GREEN}<INFERENCE_JSON_DIR>"
elif [[ -z $ONLY_GATHER ]] && ([[ -z $1 ]] || [[ -z $2 ]] || [[ -z $3 ]]); then
    echo -en "Usage: [DRY_RUN=1] [ONLY_GATHER=1] [ONLY_EVAL=1] ./eval_suite.sh ${GREEN}<INFERENCE_JSON_DIR> <JSON_FILE_NAME> <SPLIT> "
    echo -en "[<IMAGE_B64_TSV_PATH>] [<MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT>] [<POST_PROCESS_MULTI_CANDIDATES_SCRIPT>]"
    echo -en "JSON_FILE_NAME is not used, use any string like 'xxx' for it."
    exit 1
fi
if [[ -n $4 ]]; then
    IMAGE_B64_TSV_PATH=$4
fi
if [[ -n $5 ]]; then
    MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT=$5
fi
if [[ -n $6 ]]; then
    POST_PROCESS_MULTI_CANDIDATES_SCRIPT=$6
fi
if [[ ! -d $1 ]]; then
    echo -e "${RED}Error: ${NC}Directory ${GREEN}${1}${NC} for ${GREEN}INFERENCE_JSON_DIR${NC} does not exist."
    exit 1
fi
if [[ ! -f $IMAGE_B64_TSV_PATH ]]; then
    if [[ -z $SKIP_CLIP_RECALL ]] && [[ -z $ONLY_GATHER ]]; then
        echo -e "${RED}Error: ${NC}File ${GREEN}${IMAGE_B64_TSV_PATH}${NC} for ${GREEN}IMAGE_B64_TSV_PATH${NC} does not exist. Turn on ${GREEN}SKIP_CLIP_RECALL${NC} if you don't need it."
        exit 1
    fi
    echo -e "${RED}Warning: ${NC}File ${GREEN}${IMAGE_B64_TSV_PATH}${NC} for ${GREEN}IMAGE_B64_TSV_PATH${NC} does not exist. ${GREEN}SKIP_CLIP_RECALL${NC} is on."
fi
if [[ ! -f $MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT ]]; then
    echo -e "${RED}Error: ${NC}File ${GREEN}${MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT}${NC} for ${GREEN}MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT${NC} does not exist."
    exit 1
fi

if [[ -z $ONLY_GATHER ]]; then
    echo -e "Checking vdtk ..."
    vdtk > /dev/null 2>&1
else
    echo -e "ONLY_GATHER is on, skip checking vdtk ..."
fi
if [[ $? -ne 0 ]]; then
    if [[ -z $ONLY_GATHER ]]; then
        echo -e "${RED}Error: ${NC}vdtk is not installed. Please install it first."
        echo -e "\tCheck ${LIGHT_GREEN}docs/EVAL.md${NC} for details as we use a ${GREEN}modified version of vdtk${NC}"
        exit 1
    fi
    echo -e "${RED}Warning: ${NC}vdtk is not installed. ${GREEN}ONLY_GATHER${NC} is on."
fi

set -e
INFERENCE_JSON_DIR=$1
JSON_FILE_NAME=$2
SPLIT=$3
IMAGE_B64_TSV_PATH=$IMAGE_B64_TSV_PATH
MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT=$MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT
POST_PROCESS_MULTI_CANDIDATES_SCRIPT=$POST_PROCESS_MULTI_CANDIDATES_SCRIPT
GATHERED_CSV_FILE_DIR="${INFERENCE_JSON_DIR}/gathered_csv_files"

echo -e "${LIGHT_GREEN}Parameters${NC}:"
echo -e "\t${GREEN}INFERENCE_JSON_DIR${NC}: ${INFERENCE_JSON_DIR}"
echo -e "\t${GREEN}JSON_FILE_NAME${NC}: ${JSON_FILE_NAME}"
echo -e "\t${GREEN}SPLIT${NC}: ${SPLIT}"
echo -e "\t${GREEN}IMAGE_B64_TSV_PATH${NC}: ${IMAGE_B64_TSV_PATH}"
echo -e "\t${GREEN}MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT${NC}: ${MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT}"
echo -e "\t${GREEN}POST_PROCESS_MULTI_CANDIDATES_SCRIPT${NC}: ${POST_PROCESS_MULTI_CANDIDATES_SCRIPT}"
echo -e "\t${GREEN}GATHERED_CSV_FILE_DIR${NC}: ${GATHERED_CSV_FILE_DIR}"


# Parse each job in the experiment directory
echo -e "${RED}JSON_FILE_NAME is not used${NC}"
mapfile -t PRED_JSON_LS < <(find "$INFERENCE_JSON_DIR" -path '*/infer/*' -name 'infer-*.json' | sort -r)
echo -e "${GREEN}Files to be processed${NC}:" "${PRED_JSON_LS[@]}"

if [[ -n "$DEBUG" ]]; then
    breakpoint
fi

if [[ ! -z $DRY_RUN ]]; then
    echo -e "${LIGHT_GREEN}Dry run, exiting..."
    exit 0
fi

# Post-process multi candidates to one for vdtk
if [[ ! -z $ONLY_GATHER ]]; then
    echo "Only gather, skiping post processing for vdtk..."
elif [[ ! -z $NO_POST_PROCESS ]]; then
    echo "No post process, skiping post processing for vdtk..."
else
    # NOTE: replace gt based on dataset used in the `train.py`` code
    for PRED_JSON_PATH in "${PRED_JSON_LS[@]}"; do
        echo "Replace gt references to the original ones: $PRED_JSON_PATH"
        INFERENCE_JSON_DIR_=$(dirname $(dirname "$PRED_JSON_PATH"))
        echo python scripts/tools/replace_references_in_json_for_vdtk.py \
            --config-dir="$INFERENCE_JSON_DIR_/.hydra/" \
            --config-name=config \
            training.output_dir="$INFERENCE_JSON_DIR_" 
        python scripts/tools/replace_references_in_json_for_vdtk.py \
            --config-dir="$INFERENCE_JSON_DIR_/.hydra/" \
            --config-name=config \
            training.output_dir="$INFERENCE_JSON_DIR_"
    done
    if [[ -n "$DEBUG" ]]; then
        breakpoint
    fi
    for PRED_JSON_PATH in "${PRED_JSON_LS[@]}"; do
        REPLACED_GT_JSON_PATH=$(echo "$PRED_JSON_PATH" | sed 's|/infer/|/infer-post_processed/|') 
        POST_PROCESSED_PRED_JSON_PATH="${REPLACED_GT_JSON_PATH}.post.json"
        echo "Remove multiple candidates for vdtk: $PRED_JSON_PATH"
        echo python "$POST_PROCESS_MULTI_CANDIDATES_SCRIPT" -i "$REPLACED_GT_JSON_PATH" -o "$POST_PROCESSED_PRED_JSON_PATH"
        python "$POST_PROCESS_MULTI_CANDIDATES_SCRIPT" -i "$REPLACED_GT_JSON_PATH" -o "$POST_PROCESSED_PRED_JSON_PATH"
done
fi

if [[ -n "$DEBUG" ]]; then
    breakpoint
fi

# Run vdtk for each job
function metric_func()
{
    local EVAL_TYPE=$1
    local PRED_JSON_PATH=$2
    local WORK_DIR=$3
    if [[ ! -d $WORK_DIR ]]; then
        mkdir -p $WORK_DIR
    fi
    local LOG_PATH="${WORK_DIR}/${EVAL_TYPE}.log"
    if [[ -f $LOG_PATH ]]; then
        echo "Log file exists, removing: $LOG_PATH"
        rm "$LOG_PATH"
    fi
    
    if [[ $EVAL_TYPE = "scores" ]]; then
        for SCORE_TYPE in ciderd meteor rouge spice bleu; do
            # Override the log path
            LOG_PATH="${WORK_DIR}/${EVAL_TYPE}-${SCORE_TYPE}.log"
            if [[ -f $LOG_PATH ]]; then
                echo "Log file exists, removing: $LOG_PATH"
                rm "$LOG_PATH"
            fi
            echo "Time: $(date +"%m%d%y-%H%M%S")" | tee -a "${LOG_PATH}"
            echo vdtk score "${SCORE_TYPE}" "${PRED_JSON_PATH}" --split $SPLIT --save-dist-plot --save-scores | tee -a "${LOG_PATH}"
            COLUMNS=$COLUMNS_FOR_VDTK vdtk score "${SCORE_TYPE}" "${PRED_JSON_PATH}" --split $SPLIT --save-dist-plot --save-scores | tee -a "${LOG_PATH}"
            echo "Time: $(date +"%m%d%y-%H%M%S")" | tee -a "${LOG_PATH}"
        done

    elif [[ $EVAL_TYPE = "content" ]]; then
        echo "Time: $(date +"%m%d%y-%H%M%S")" | tee -a "${LOG_PATH}"
        echo vdtk content-recall "${PRED_JSON_PATH}" --split $SPLIT | tee -a "${LOG_PATH}"
        COLUMNS=$COLUMNS_FOR_VDTK vdtk content-recall "${PRED_JSON_PATH}" --split $SPLIT | tee -a "${LOG_PATH}"
        echo "Time: $(date +"%m%d%y-%H%M%S")" | tee -a "${LOG_PATH}"

    elif [[ $EVAL_TYPE = "clip" ]]; then
        if [[ -n $SKIP_CLIP_RECALL ]]; then
            echo "Skip clip recall, exiting..."
        else
        JSON_DIR=$(dirname "$PRED_JSON_PATH")
        local MERGED_JSON_PATH="${JSON_DIR}/pred-media_b64.json"

        echo "Time: $(date +"%m%d%y-%H%M%S")" | tee -a "${LOG_PATH}"
        echo python $MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT \
        -t "${IMAGE_B64_TSV_PATH}" \
        -j "${PRED_JSON_PATH}" \
        -o "${MERGED_JSON_PATH}" | tee -a "${LOG_PATH}"
        python $MERGE_TSV_INTO_JSON_FOR_VDTK_SCRIPT \
        -t "${IMAGE_B64_TSV_PATH}" \
        -j "${PRED_JSON_PATH}" \
        -o "${MERGED_JSON_PATH}"
        du -sh "${MERGED_JSON_PATH}" | tee -a "${LOG_PATH}"

        echo "Time: $(date +"%m%d%y-%H%M%S")" | tee -a "${LOG_PATH}"
        echo vdtk clip-recall "${MERGED_JSON_PATH}" --split $SPLIT | tee -a "${LOG_PATH}"
        COLUMNS=$COLUMNS_FOR_VDTK vdtk clip-recall "${MERGED_JSON_PATH}" --split $SPLIT | tee -a "${LOG_PATH}"
        echo "Time: $(date +"%m%d%y-%H%M%S")" | tee -a "${LOG_PATH}"

        if [[ -f $MERGED_JSON_PATH ]]; then
            echo "Removing: $MERGED_JSON_PATH"
            rm "$MERGED_JSON_PATH"
        fi

        fi
    else
        echo "Unknown EVAL_TYPE: $EVAL_TYPE"
        exit 1
    fi
}

if [[ ! -z $ONLY_GATHER ]]; then
    echo "Only gather, skip evaluation..."
else
echo "Evaluating ..."
for PRED_JSON_PATH in "${PRED_JSON_LS[@]}"; do
    echo "Precessing: $PRED_JSON_PATH"
    
    METRIC_DIR=$(dirname "$PRED_JSON_PATH")/metrics/$(basename "$PRED_JSON_PATH" .json)
    if [[ -z $NO_POST_PROCESS ]]; then
        REPLACED_GT_JSON_PATH=$(echo "$PRED_JSON_PATH" | sed 's|/infer/|/infer-post_processed/|') 
        POST_PROCESSED_PRED_JSON_PATH="${REPLACED_GT_JSON_PATH}.post.json"
    else
        POST_PROCESSED_PRED_JSON_PATH=$PRED_JSON_PATH
    fi

    if [[ -n "$DEBUG" ]]; then
        breakpoint
    fi

    for EVAL_TYPE in content scores clip; do
        metric_func "${EVAL_TYPE}" "${POST_PROCESSED_PRED_JSON_PATH}" "$METRIC_DIR" &  # Parallel
    done
    wait
done
fi
if [[ -n "$DEBUG" ]]; then
    breakpoint
fi
# Gather all the csv files from each job
echo -e "${RED}JSON_FILE_NAME is not used${NC}"
mapfile -t PRED_JSON_LS < <(find "$INFERENCE_JSON_DIR" -path '*/infer/*' -name 'infer-*.json' | sort -r)
echo "files to be processed:" "${PRED_JSON_LS[@]}"

if [[ -n "$DEBUG" ]]; then
    breakpoint
fi

if [[ ! -z $DRY_RUN ]]; then
    echo "Dry run, exiting..."
    exit 0
fi

function process_file(){
    local LOG_PATH=$1
    local CSV_PATH=$2
    grep -E '^┃|│' $LOG_PATH | awk -F '[┃│]' -v log_path="$LOG_PATH" 'BEGIN{OFS=","} /Dataset/{header="Log Path,"$2","$3","$4","$5","$6","$7","$8; gsub(/^[[:blank:]]+|[[:blank:]]+$/, "", header); gsub(/,$/, "", header); print header}
    /\.json/{row=log_path","$2","$3","$4","$5","$6","$7","$8; gsub(/^[[:blank:]]+|[[:blank:]]+$/, "", row); gsub(/,$/, "", row); print row}' | sed 's/ //g' > $CSV_PATH
}

function parse_to_csv()
{
    local EVAL_TYPE=$1
    local PRED_JSON_PATH=$2
    local WORK_DIR=$3

    if [[ ! -d $WORK_DIR ]]; then
        mkdir -p $WORK_DIR
    fi
    local LOG_PATH="${WORK_DIR}/${EVAL_TYPE}.log"
    local CSV_PATH="${WORK_DIR}/${EVAL_TYPE}.csv"
    if [[ -f $CSV_PATH ]]; then
        echo "CSV file exists, removing: $CSV_PATH"
        rm $CSV_PATH
    fi
    
    if [[ $EVAL_TYPE = "scores" ]]; then
        for SCORE_TYPE in bleu meteor rouge spice ciderd; do
            # Override the log path
            local LOG_PATH="${WORK_DIR}/${EVAL_TYPE}-${SCORE_TYPE}.log"
            local CSV_PATH="${WORK_DIR}/${EVAL_TYPE}-${SCORE_TYPE}.csv"
            echo "Convert log to csv: $LOG_PATH to $CSV_PATH"
            process_file $LOG_PATH $CSV_PATH
        done
    elif [[ $EVAL_TYPE = "clip" ]] || [[ $EVAL_TYPE = "content" ]]; then
        process_file $LOG_PATH $CSV_PATH
    else
        echo "Unknown EVAL_TYPE: $EVAL_TYPE"
        exit 1
    fi
}



if [[ ! -z $ONLY_EVAL ]]; then
    echo "Only eval, exiting..."
    exit 0
else
echo "Gathering ..."

for PRED_JSON_PATH in "${PRED_JSON_LS[@]}"; do
    echo "Convert log to csv: $PRED_JSON_PATH"

    METRIC_DIR=$(dirname "$PRED_JSON_PATH")/metrics/$(basename "$PRED_JSON_PATH" .json)
    for EVAL_TYPE in scores content clip; do
        parse_to_csv "${EVAL_TYPE}" "${PRED_JSON_PATH}" "$METRIC_DIR" &
    done
    wait
done

# remove auxiliary `gathered_csv_files` directory if exists
find $INFERENCE_JSON_DIR -path '*/gathered_csv_files' | xargs rm -rf
if [[ ! -d $GATHERED_CSV_FILE_DIR ]]; then
    mkdir -p $GATHERED_CSV_FILE_DIR
fi

_ONE_PRED_JSON_FILE="${PRED_JSON_LS[0]}"
_METRIC_CSV_DIR="$(dirname $_ONE_PRED_JSON_FILE)"
mapfile -t CSV_METRIC_NAME_LS < <(find $_METRIC_CSV_DIR -path '*/metrics/*' -name '*.csv' -exec basename {} \; | sort -r | uniq)
echo "CSV files to be processed:" "${CSV_METRIC_NAME_LS[@]}"

if [[ -n "$DEBUG" ]]; then
    breakpoint
fi

_INFERENCE_JSON_DIR_NAME="$(basename $INFERENCE_JSON_DIR)"
for CSV_METRIC_NAME in "${CSV_METRIC_NAME_LS[@]}"; do
    # NOTE: find all the *.csv file with certain name and copy them into one file in GATHERED_CSV_FILE_DIR
    echo "Gathering: $CSV_METRIC_NAME"
    # GATHERED_CSV_FILE_PATH="${GATHERED_CSV_FILE_DIR}/${_INFERENCE_JSON_DIR_NAME}-${CSV_METRIC_NAME}"
    GATHERED_CSV_FILE_PATH="${GATHERED_CSV_FILE_DIR}/${CSV_METRIC_NAME}"
if [[ -n "$DEBUG" ]]; then
    echo "in line"
    breakpoint
fi
    if [[ -f $GATHERED_CSV_FILE_PATH ]]; then
        echo "CSV file exists, removing: $GATHERED_CSV_FILE_PATH"
        rm $GATHERED_CSV_FILE_PATH
    fi
if [[ -n "$DEBUG" ]]; then
    echo "in line"
    breakpoint
fi
    for SRC_FILE in $(find $INFERENCE_JSON_DIR -not -path "$GATHERED_CSV_FILE_DIR/*" -name "${CSV_METRIC_NAME}" | sort -r); do 
        cat "$SRC_FILE" >> "${GATHERED_CSV_FILE_PATH}"
    done
    # awk -F, -v prefix="$INFERENCE_JSON_DIR" -v suffix="/[^/]*\\.log$" 'BEGIN{OFS=","} NR==1{print "Line Number", $1, $0; next} {original=$1; gsub(prefix, "", $1); original=$1; gsub(suffix, "", $1); print NR, original, $0}' "${GATHERED_CSV_FILE_PATH}".tmp > "${GATHERED_CSV_FILE_PATH}"
    awk -i inplace -F, -v prefix="$INFERENCE_JSON_DIR" -v suffix="/metrics/[^/]*\\.log$" 'BEGIN{OFS=","} {original=$1; gsub(prefix, "", $1); gsub(suffix, "", $1); print original, $0}' "${GATHERED_CSV_FILE_PATH}"
    awk -i inplace '!seen[$0]++' "${GATHERED_CSV_FILE_PATH}"
    awk -i inplace '!seen[$0]++' "${GATHERED_CSV_FILE_PATH}"
    # NOTE: log_path, log_name, the json file name, the metric name, the metrics, ...
    if [[ $CSV_METRIC_NAME = *"clip"* ]]; then
        awk -i inplace 'NR<2{print $0;next}{print $0 | "sort -t , -k 3,3 -k 2,2"}' "${GATHERED_CSV_FILE_PATH}"
        # NOTE: remove the repeated reference row
        # awk -i inplace '1;/(reference)/{exit} ' "${GATHERED_CSV_FILE_PATH}"
    else
        awk -i inplace 'NR<2{print $0;next}{print $0 | "sort -t , -k 2,2"}' "${GATHERED_CSV_FILE_PATH}"
    fi
done


find $GATHERED_CSV_FILE_DIR -name "*.csv" -exec \
python -c "import pandas as pd;import sys; file=sys.argv[1]; df=pd.read_csv(file); df.to_csv(file, index=False); df.to_excel(file+'.xlsx', index=False)" {} \;

ALL_GATHERED_CSV_FILE_PATH="${GATHERED_CSV_FILE_DIR}/${_INFERENCE_JSON_DIR_NAME}-all.csv"
find $GATHERED_CSV_FILE_DIR -name "*.csv" -not -path "$ALL_GATHERED_CSV_FILE_PATH" | sort -r | xargs cat > "${ALL_GATHERED_CSV_FILE_PATH}"
fi

ALL_GATHERED_XLSX_FILE_PATH="${GATHERED_CSV_FILE_DIR}/${_INFERENCE_JSON_DIR_NAME}-all.csv.xlsx"
if [[ -f $ALL_GATHERED_XLSX_FILE_PATH ]]; then
    echo "XLSX file exists, removing: $ALL_GATHERED_XLSX_FILE_PATH"
    rm $ALL_GATHERED_XLSX_FILE_PATH
fi

echo "Formating: $ALL_GATHERED_XLSX_FILE_PATH"
find $GATHERED_CSV_FILE_DIR -name "*.csv.xlsx" -not -name "$ALL_GATHERED_XLSX_FILE_PATH" -exec \
python -c \
"import pandas as pd;
import sys;
import os.path as osp;
mode='w' if not osp.exists('${ALL_GATHERED_XLSX_FILE_PATH}') else 'a';
with pd.ExcelWriter('${ALL_GATHERED_XLSX_FILE_PATH}', mode=mode) as writer:  \
df=pd.read_excel(sys.argv[1]);  \
df.to_excel(writer, sheet_name=osp.basename(sys.argv[1]), index=False);
" {} \;

python scripts/tools/merge_sheets_xlsx.py "$ALL_GATHERED_XLSX_FILE_PATH"
