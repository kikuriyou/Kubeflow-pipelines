#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import shutil
import argparse
from datetime import datetime, date, timedelta, timezone
import json
import warnings
from io import StringIO
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim


"""認証キーを読み込み"""
print('loading credentials')
credentials = service_account.Credentials.from_service_account_info(
    {
        """使用するGCPプロジェクトの認証キー情報を入れる"""
    },
)

DEFAULT_PARAMS = {
    'num_topics': 6,
    'chunk_size': 1000,
    'num_pass': 30,
    'workers': 3
}


def get_current_time(area='JST'):
    return datetime.now(
        timezone(timedelta(hours=+9), area)
    ).strftime('%Y-%m-%d %H:%M:%S')


def get_prev_date(days):
    today = datetime.today()
    prev_date = today - timedelta(days=days)
    prev_date_str = datetime.strftime(prev_date, '%Y-%m-%d')
    return prev_date_str


def parse_arguments():
    """Parse job arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preprocess_output',
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
        '--prev_date',
        help='Previous date (used for loading the model)'
    )
    parser.add_argument(
        '--date',
        help='Date'
    )
    parser.add_argument(
        '--dict_file',
        help='Dictionary file name (w/o ".csv")',
        default='dict'
    )
    parser.add_argument(
        '--dataset_file',
        help='Dataset file name (w/o ".csv")',
        default='dataset'
    )
    parser.add_argument(
        '--tmp_dir',
        help='Directory for temporal result files',
        required=True
    )
    parser.add_argument(
        '--learning_type',
        help='Reset or update the model [ "reset" | "update" ]',
        default='update'
    )
    parser.add_argument(
        '--pipeline_version',
        help='Pipeline version'
    )
    parser.add_argument(
        '--output',
        help='Output directory',
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    params = DEFAULT_PARAMS
    params.update({k: arg for k, arg in arguments.items() if arg is not None})

    # データセット日時のチェック
    if params['date'] == '':
        params['date'] = get_prev_date(1)
    else:
        pass

    # モデル更新時のチェック
    if (params['learning_type'] == 'update') & (params['prev_date'] == ''):
        print('When updating the model, you need "--prev_date" argument.')
        sys.exit()

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


def load_gcs(project_name, bucket_name, local_file, gcs_file, mode):
    """
    - インスタンス上のファイルをGCSにアップロード
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


def get_dict(args):
    """全ワードのデータセットを読み込み、dictionaryを生成"""
    print('Generating dictionary....')

    cols = ['names']
    gcs_csv = os.path.join(args['preprocess_output'], args['dict_file'] + '.csv')
    data_raw = read_gcs(args['project'], args['bucket'], gcs_csv, cols)
    
    words = data_raw.values.tolist()
    dict_word = corpora.Dictionary(words)

    return dict_word


def get_word(dict_word, args):
    """
    デッキのデータセットを読み込む
    - data_word: 使用ワードのデータセット
    - data_uid: data_wordに紐づくidのリスト
    - corpus_word: data_wordのコーパス
    """
    print('Loading dataset....')

    # csvファイルを読み込む
    cols = ['id', 'hero0', 'hero1', 'hero2', 'hero3']
    gcs_csv = os.path.join(args['preprocess_output'], args['dataset_file'] + '.csv')
    data_raw = read_gcs(args['project'], args['bucket'], gcs_csv, cols)

    # idのリスト, デッキ内容のリストに分ける
    data_uid = data_raw['id']
    data_word = data_raw.drop('id', axis=1)

    # 全ワードのdictionaryを参照しながらコーパスに変換
    words = data_word.values.tolist()
    corpus_word = [dict_word.doc2bow(deck) for deck in decks]

    return data_deck, data_uid, corpus_deck


def main(args):
    """LDAでデッキごとにトピック番号を割り当てる"""
    # 実行時刻を取得
    execution_time = get_current_time()

    # 全ワードのマスターデータを読み込みdictionaryを取得
    dict_deck = get_dict(args)

    # 使用ワードのデータセットを読み込む
    data_deck, data_uid, corpus_deck = get_deck(dict_deck, args)

    # ディレクトリを指定
    if not os.path.isdir(args['tmp_dir']):
        os.mkdir(args['tmp_dir'])
    model_file = os.path.join(args['tmp_dir'], 'model')

    PREV_MODEL_DIR = os.path.join(args['output'], 'workflow_' + args['prev_date'], 'model')
    MODEL_DIR = os.path.join(args['output'], 'workflow_' + args['date'], 'model')
    gcs_file_prev = os.path.join(PREV_MODEL_DIR, 'model')
    gcs_file_new = os.path.join(MODEL_DIR, 'model')
    suffixes = ['', '.expElogbeta.npy', '.id2word', '.state']

    # LDAモデルを学習
    if args['learning_type'] == 'reset':
        print('Running the model....')

        # モデルを学習（リセット）
        lda = models.ldamulticore.LdaMulticore(
            corpus=corpus_deck, 
            workers=args['workers'], 
            id2word=dict_deck, 
            num_topics=args['num_topics'], 
            chunksize=args['chunk_size'], 
            passes=args['num_pass'], 
            minimum_probability=0., 
            random_state=1
        )
        lda.save(model_file)

        # 学習済みモデルをGCSにアップロード
        for suffix in suffixes:
            load_gcs(args['project'], args['bucket'], model_file + suffix, gcs_file_new + suffix, 'upload')

    elif args['learning_type'] == 'update':
        print('Updating the model....')
        
        # 前日のモデルをGCSからダウンロード
        for suffix in suffixes:
            load_gcs(args['project'], args['bucket'], model_file + suffix, gcs_file_prev + suffix, 'download')

        # モデル更新
        lda = models.LdaModel.load(model_file)
        lda.update(corpus_deck)
        lda.save(model_file)

        # 更新済みモデルをGCSにアップロード
        for suffix in suffixes:
            load_gcs(args['project'], args['bucket'], model_file + suffix, gcs_file_new + suffix, 'upload')
    

    # 保存先
    OUTPUT_DIR = os.path.join(args['output'], 'workflow_' + args['date'], 'train')
    
    # pyLDAvisでトピック情報を可視化
    print('Saving pyLDAvis file....')

    # ディレクトリを指定
    if not os.path.isdir(args['tmp_dir']):
        os.mkdir(args['tmp_dir'])
    vis_file = os.path.join(args['tmp_dir'], 'pyLDAvis.html')
    gcs_file = os.path.join(OUTPUT_DIR, 'pyLDAvis.html')
    
    # pyLDAvisを出力
    vis = pyLDAvis.gensim.prepare(lda, corpus_deck, dict_deck)
    pyLDAvis.save_html(vis, vis_file)
    
    # GCSにアップロード
    load_gcs(args['project'], args['bucket'], vis_file, gcs_file, 'upload')


    # topicNoを結合
    print('Concatenating dataset and allocated topic distribution....')
    data_deck_topic = data_deck.copy()
    topic_cols = ['topic{}'.format(i) for i in range(args['num_topics'])]
    topic_prob = np.array([[y for (x,y) in lda[corpus_deck[i]]] for i in range(len(corpus_deck))])
    for i, topic_col in enumerate(topic_cols):
        data_deck_topic[topic_col] = topic_prob[:, i]

    # カラムを並べ替え
    print('Saving result file....')
    data_topic = pd.concat([data_uid, data_deck_topic], axis=1)
    cols = ['date']
    cols.extend(data_topic.columns)
    data_topic['date'] = args['date']
    data_topic = data_topic.loc[:, cols]
    data_topic['execution_time'] = execution_time
    data_topic['version'] = args['pipeline_version']

    # csvを保存
    if not os.path.isdir(args['tmp_dir']):
        os.mkdir(args['tmp_dir'])
    res_file = os.path.join(args['tmp_dir'], '{}.csv'.format(args['table']))
    gcs_file = os.path.join(OUTPUT_DIR, '{}.csv'.format(args['table']))
    data_topic.to_csv(res_file, header=False, index=False)

    # GCSにアップロード
    load_gcs(args['project'], args['bucket'], res_file, gcs_file, 'upload')


    # output
    try:
        with open('/output.txt', 'w') as f:
            f.write(OUTPUT_DIR)
    except:
        pass


    print('Training done.')


if __name__ == '__main__':
    job_args = parse_arguments()
    main(job_args)


"""
TODO:
- あとでクラスタリングも加える
"""