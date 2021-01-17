import boto3
import os
from botocore.exceptions import ClientError
import ast
import pandas as pd

access_key = 'VN2M29BH4PCAJT9ABZKN'
secret_access_key = 'O9bcBpaprrXr6Q3dorn0XYI4Kp8go6oBDBYFYqeD'
endpoint = 'http://s3.dev.obdc.bcs.bloomberg.com'
os.environ['AWS_ACCESS_KEY_ID'] = access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_access_key
os.environ['AWS_ENDPOINT'] = endpoint

import logging
from scripts.dsp.dsp_utils import data_io_s3
service = data_io_s3.connect_to_s3()

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

def download_bucket(remote_directory_name, download_dir=None, bucket_name='aspangher', s3_service=service):
    bucket = s3_service.Bucket(bucket_name)
    depth = sum(map(lambda x: x == '/', remote_directory_name))

    if not download_dir:
        download_dir = remote_directory_name

    for obj in bucket.objects.filter(Prefix=remote_directory_name):
        logging.info(obj)
        full_path = splitall(obj.key)
        dir_path = os.path.join(download_dir, *full_path[depth: -1])
        filename = full_path[-1]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        ## output
        output_path = os.path.join(dir_path, filename)
        bucket.download_file(obj.key, output_path)

def upload_dir(root_path, bucket_name='aspangher', s3_service=service):
    try:
        my_bucket = s3_service.Bucket(bucket_name)
        for path, subdirs, files in os.walk(root_path):
            path = path.replace("\\", "/")
            for file in files:
                my_bucket.upload_file(os.path.join(path, file), path + '/' + file)
                logging.info('uploaded %s...' % file)

    except Exception as err:
        logging.info(err)


def download_file(file_name, object_name=None, bucket='aspangher', s3_service=service):
    if file_name is None: ## useful for concise code in model_runner.py
        return

    bucket = s3_service.Bucket(bucket)
    if object_name is None:
        object_name = file_name
    try:
        response = bucket.download_file(object_name, file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_file(file_name, object_name=None, bucket='aspangher', s3_service=service):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    bucket = s3_service.Bucket(bucket)
    if object_name is None:
        object_name = file_name
    try:
        response = bucket.upload_file(file_name, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def ls(dir_name, bucket='aspangher', s3_service=service):
    b = s3_service.Bucket(bucket)
    return list(b.objects.filter(Prefix=dir_name))


# mappers
def get_log_lines(s3_obj):
    log_str = s3_obj.get()['Body'].read().decode('utf-8')
    return log_str.split('\n')

def split_eval_lines(log_lines=None,  s3_obj=None):
    if not log_lines:
        log_lines = get_log_lines(s3_obj)
    ## eval lines
    search_term = "{'eval_loss"
    eval_lines = list(filter(lambda x: search_term in x, log_lines))
    eval_lines = list(map(lambda x: search_term + x.split(search_term)[1], eval_lines))
    eval_dicts = list(map(ast.literal_eval, eval_lines))
    if 'eval_classification_report' in eval_dicts[0]:
        list(map(lambda x: x.update(x.pop('eval_classification_report')), eval_dicts))
    eval_df = pd.DataFrame(eval_dicts)
    return eval_df

def split_param_lines(log_lines=None,  s3_obj=None):
    if not log_lines:
        log_lines = get_log_lines(s3_obj)
    ## parse param lines
    params_lines = []
    in_params = False
    for line in log_lines:
        if in_params:
            params_lines.append(line)
        if 'MODEL PARAMS:' in line:
            in_params = True
        if line == '}':
            in_params = False
    params_lines = params_lines[1:]
    params_lines = ['{'] + params_lines
    param_str = ' '.join(params_lines).replace('true', 'True').replace('false', 'False').replace('null', 'None')
    param_dict = ast.literal_eval(param_str)
    return param_dict

