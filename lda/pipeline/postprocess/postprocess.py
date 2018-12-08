#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
from datetime import datetime, date, timedelta
from io import StringIO
import pandas as pd
import pandas_gbq
from google.cloud import storage
from google.oauth2 import service_account


"""認証キーを読み込み"""
credentials = service_account.Credentials.from_service_account_info(
    {
        """使用するGCPプロジェクトの認証キー情報を入れる"""
    },
)


def get_prev_date(days):
    today = datetime.today()
    prev_date = today - timedelta(days=days)
    prev_date_str = datetime.strftime(prev_date, '%Y-%m-%d')
    return prev_date_str


def parse_arguments():
    """Parse job arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training_output',
        help='Preprocessed output',
        required=True
    )
    parser.add_argument(
        '--project',
        help='GCP project ID',
        required=True
    )
    parser.add_argument(
        '--bucket',
        help='GCS bucket name',
        required=True
    )
    parser.add_argument(
        '--table',
        help='Table name',
        required=True
    )
    parser.add_argument(
        '--date',
        help='Date',
        required=True
    )
    parser.add_argument(
        '--output',
        help='Output directory',
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__
    params = {k: arg for k, arg in arguments.items() if arg is not None}

    if params['date'] == '':
        params['date'] = get_prev_date(1)
    else:
        pass

    return params


def read_gcs(project_name, bucket_name, gcs_csv, cols):
    """
    - GCS上のテキストファイルを読み込み, pandas dataframeで返す
    (Input)
    project_name: GCS project ID
    bucket_name:  GCS bucket name
    gcs_csv:      GCS file name
    cols:         Column names of dataframe
    """
    client = storage.Client(project_name, credentials=credentials)
    bucket = client.get_bucket(bucket_name)

    # CSVを取得
    blob = storage.Blob(gcs_csv.replace('gs://{}/'.format(bucket_name), ''), bucket)
    content = blob.download_as_string()
    s = str(content, 'utf-8')
    data = StringIO(s)

    df = pd.read_csv(data, names=cols)

    return df


def main(args):
    """LDAの結果をBigQueryのテーブルにアップロード"""

    # GCSからダウンロードして読み込み
    print('Downloading results from GCS....')
    gcs_csv = os.path.join(args['training_output'], '{}.csv'.format(args['table']))
    cols = [
        'date', 'id', 
        'name0', 'name1', 'name2', 'name3', 
        'topic0', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5',
        'execution_time', 'pipeline_version'
    ]
    df = read_gcs(args['project'], args['bucket'], gcs_csv, cols).astype({
        'date': 'object', 'id': 'object',
        'name0': 'object', 'name1': 'object', 'name2': 'object', 'name3': 'object', 
        'topic0': 'float32', 'topic1': 'float32', 'topic2': 'float32', 
        'topic3': 'float32', 'topic4': 'float32', 'topic5': 'float32',
        'execution_time': 'object', 'pipeline_version': 'object'
    })

    # テーブルに追加
    print('Uploading results to BigQuery....')
    destination_table = 'WORK.{}'.format(args['table'])
    pandas_gbq.to_gbq(df, destination_table, args['project'], if_exists='append', credentials=credentials)

    # 保存先
    OUTPUT_DIR = os.path.join(args['output'], 'workflow_' + args['date'], 'postprocess')
    
    # output
    try:
        with open('/output.txt', 'w') as f:
            f.write(OUTPUT_DIR)
    except:
        pass


    print('Postprocessing done.')


if __name__ == '__main__':
    job_args = parse_arguments()
    main(job_args)
