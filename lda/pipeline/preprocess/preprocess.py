#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
from datetime import datetime, date, timedelta
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
        '--date',
        help='Date',
        required=True
    )
    parser.add_argument(
        '--dict_file',
        help='Dictionary csv file',
        default='dict'
    )
    parser.add_argument(
        '--dataset_file',
        help='Dataset csv file',
        default='dataset'
    )
    parser.add_argument(
        '--tmp_dir',
        help='Temporal directory',
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


def load_gcs(project_name, bucket_name, local_file, gcs_file, mode):
    """
    - インスタンス上のファイルをGCSにアップロード/ダウンロード
    (Input)
    project_name: GCS project ID
    bucket_name:  GCS bucket name
    local_file:   File name located in the instance
    gcs_file:     GCS file name
    mode:         [ 'upload' | 'download' ]
    """
    client = storage.Client(project_name, credentials=credentials)
    bucket = client.get_bucket(bucket_name)
    
    # ファイルをアップロード/ダウンロード
    blob = bucket.blob(gcs_file.replace('gs://{}/'.format(bucket_name), ''))
    if mode == 'upload':
        blob.upload_from_filename(local_file)
    elif mode == 'download':
        blob.download_to_filename(local_file)


def main(args):
    """dictionaryを読込"""
    print('Loading dictionary data from BigQuery....')

    # BigQueryから引っ張ってくる
    query = """SELECT * FROM SAMPLE.NAMES"""
    df = pandas_gbq.read_gbq(query, project_id=args['project'], dialect='standard', credentials=credentials)

    # 保存先の指定
    OUTPUT_DIR = os.path.join(args['output'], 'workflow_' + args['date'], 'preprocess')

    # ファイル名の指定
    if not os.path.isdir(args['tmp_dir']):
        os.mkdir(args['tmp_dir'])
    local_file = os.path.join(args['tmp_dir'], args['dict_file'] + '.csv')
    gcs_file = os.path.join(OUTPUT_DIR, args['dict_file'] + '.csv')

    # csvを保存
    df.to_csv(local_file, header=False, index=False)

    # GCSにアップロード
    load_gcs(args['project'], args['bucket'], local_file, gcs_file, 'upload')

    """データセットを読込"""
    print('Loading dataset from BigQuery....')

    # BigQueryから引っ張ってくる
    query = """SELECT * FROM SAMPLE.DUMMY"""
    df = pandas_gbq.read_gbq(query, args['project'], dialect='standard', credentials=credentials)

    # ディレクトリの指定
    if not os.path.isdir(args['tmp_dir']):
        os.mkdir(args['tmp_dir'])
    local_file = os.path.join(args['tmp_dir'], args['dataset_file'] + '.csv')
    gcs_file = os.path.join(OUTPUT_DIR, args['dataset_file'] + '.csv')

    # csvを保存
    df.to_csv(local_file, header=False, index=False)
    
    # GCSにアップロード
    load_gcs(args['project'], args['bucket'], local_file, gcs_file, 'upload')

    # output
    try:
        with open('/output.txt', 'w') as f:
            f.write(OUTPUT_DIR)
    except:
        pass

    print('Preprocessing done.')


if __name__ == '__main__':
    job_args = parse_arguments()
    main(job_args)
