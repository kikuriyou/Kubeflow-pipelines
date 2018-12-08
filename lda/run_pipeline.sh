#! /bin/bash
usage () {
    echo "usage: ./run_pipeline.sh DATE[yyyy-mm-dd]"
}

# Command Line Argument
#COMMAND=`basename $0`
#if [ $# -ne 1 ]; then
#    usage
#    exit 1;
#fi
PREV_DATE=$1
DATE=$2

# Environmental variables
PIPELINE="pipeline"
OUTPUT="output"
TMP="tmp"
MODEL="model"
DIR_EXEC="`pwd`/${PIPELINE}"
#DIR_CONTAINER="/${PIPELINE}"
DIR_CONTAINER=${DIR_EXEC}        # ローカルテスト用

# GCP settings
PROJECT="project_id"
BUCKET="bucket"
GCS_DIR="gs://${BUCKET}/kfp_sample"

# Input files
DICT_FILE="dict"
DATASET_FILE="dataset"

# BigQuery table
DATASET="WORK"
TABLE="TOPIC_RESULT"

echo ""
echo "PIPELINE:      ${PIPELINE}"
echo "STEP:          ${STEP}"
echo "DIR_EXEC:      ${DIR_EXEC}"
echo "DIR_CONTAINER: ${DIR_CONTAINER}"
echo ""


# Preprocess
python ${PIPELINE}/preprocess/preprocess.py \
    --project        ${PROJECT} \
    --bucket         ${BUCKET} \
    --date           ${DATE} \
    --tmp_dir        ${DIR_CONTAINER}/${TMP} \
    --output         ${GCS_DIR}


# train
python ${PIPELINE}/train/train.py \
    --preprocess_output  ${GCS_DIR}/workflow_${DATE}/preprocess \
    --project            ${PROJECT} \
    --bucket             ${BUCKET} \
    --table              ${TABLE} \
    --prev_date          ${PREV_DATE} \
    --date               ${DATE} \
    --tmp_dir            ${DIR_CONTAINER}/${TMP} \
    --learning_type      update \
    --pipeline_version   ${PIPELINE} \
    --output             ${GCS_DIR}


# Postprocess
python ${PIPELINE}/postprocess/postprocess.py \
    --training_output ${GCS_DIR}/workflow_${DATE}/train \
    --project         ${PROJECT} \
    --bucket          ${BUCKET} \
    --table           ${TABLE} \
    --date            ${DATE} \
    --output          ${GCS_DIR}


exit $?
