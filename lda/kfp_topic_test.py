#!/usr/bin/env python3

from datetime import datetime, date, timedelta
import kfp.dsl as dsl
import kfp.compiler as compiler

PROJECT_ID = 'project_id'
BUCKET = 'bucket'

def preprocess_op(project: 'GcpProject', bucket, date, dict_file, dataset_file, tmp_dir,
                  preprocess_output: 'GcsUri[Directory]', step_name='preprocess'):
    return dsl.ContainerOp(
        name = step_name,
        image = 'gcr.io/{}/kfp/pre:latest'.format(PROJECT_ID),
        arguments = [
            '--project',      project,
            '--bucket',       bucket,
            '--date',         date,
            '--dict_file',    dict_file,
            '--dataset_file', dataset_file,
            '--tmp_dir',      tmp_dir,
            '--output',       preprocess_output
        ],
        file_outputs = {'preprocess': '/output.txt'}
    )


def training_op(preprocess_output: 'GcsUri[Directory]', project: 'GcpProject', bucket, table, 
                prev_date, date, dict_file, dataset_file, learning_type, pipeline_version, tmp_dir, 
                training_output: 'GcsUri[Directory]', step_name='train'):
    return dsl.ContainerOp(
        name = step_name,
        image = 'gcr.io/{}/kfp/train:latest'.format(PROJECT_ID,
        arguments = [
            '--preprocess_output', preprocess_output,
            '--project',           project,
            '--bucket',            bucket,
            '--table',             table,
            '--prev_date',         prev_date,
            '--date',              date,
            '--dict_file',         dict_file,
            '--dataset_file',      dataset_file,
            '--learning_type',     learning_type,
            '--pipeline_version',  pipeline_version,
            '--tmp_dir',           tmp_dir,
            '--output',            training_output
        ],
        file_outputs = {'train': '/output.txt'}
    )


def postprocess_op(training_output: 'GcsUri[Directory]', project: 'GcpProject', bucket, table, date, 
                   postprocess_output: 'GcsUri[Directory]', step_name='postprocess'):
    return dsl.ContainerOp(
        name = step_name,
        image = 'gcr.io/{}/kfp/post:latest'.format(PROJECT_ID,
        arguments = [
            '--training_output', training_output,
            '--project',         project,
            '--bucket',          bucket,
            '--table',           table,
            '--date',            date,
            '--output',          postprocess_output
        ],
        file_outputs = {'postprocess': '/output.txt'}
    )


@dsl.pipeline(
    name='LDA pipeline',
    description='LDA pipeline running on every Wednesday'
)
def kubeflow_training(
    output:        dsl.PipelineParam, 
    project:       dsl.PipelineParam,
    bucket:        dsl.PipelineParam=dsl.PipelineParam(name='bucket',          value=bucket),
    table:         dsl.PipelineParam=dsl.PipelineParam(name='table',           value='TOPIC_TRY'),
    prev_date:     dsl.PipelineParam=dsl.PipelineParam(name='prev-date',       value=''),
    date:          dsl.PipelineParam=dsl.PipelineParam(name='date',            value=''),
    dict_file:     dsl.PipelineParam=dsl.PipelineParam(name='dictionary-file', value='dict'),
    dataset_file:  dsl.PipelineParam=dsl.PipelineParam(name='dataset-file',    value='dataset'),
    learning_type: dsl.PipelineParam=dsl.PipelineParam(name='learning-type',   value='update')):


    # TODO: use the argo job name as the workflow
    #workflow = '{{workflow.name}}'
    pipeline_version = __file__

    # Make pipeline
    preprocess = preprocess_op(project, bucket, date, dict_file, dataset_file, '/tmp', output)
    training = training_op(preprocess.output, project, bucket, table, prev_date, date, 
                           dict_file, dataset_file, learning_type, pipeline_version, '/tmp', output)
    postprocess = postprocess_op(training.output, project, bucket, table, date, output)


if __name__ == '__main__':
    compiler.Compiler().compile(kubeflow_training, __file__ + '.tar.gz')

