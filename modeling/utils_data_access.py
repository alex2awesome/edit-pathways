import os


def get_fs():
    import s3fs
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'http://s3.dev.obdc.bcs.bloomberg.com'})
    return fs


def download_model_files_bb(remote_model, use_pretrained_dir=True, local_path=None, use_zip=True):
    """
    Download pretrained model files from bb S3.

    params:
    * remote_model_path: s3 directory, (or filename, if `use_pretrained_dir`=True)
    * use_pretrained_dir: whether to look in aspangher/transformer-pretrained-models or not
    * local_path: where to download the model files. If none, default to the basename of `remote_model_path`
    * use_zip: whether to unzip the model directory or not.
        If `use_zip` is False and model_path ends in a `/`, then `fs.get()` is called recursively, otherwise, not.
    """
    if local_path is None:
        local_path = remote_model

    fs = get_fs()

    # format model name/path
    model_file_name = '%s.zip' % remote_model if use_zip else remote_model
    model_path_name = 'aspangher/transformer-pretrained-models/%s' % model_file_name if use_pretrained_dir else model_file_name

    print('downloading %s -> %s...' % (model_path_name, local_path))
    # download and optionally unzip
    if use_zip:
        fs.get(model_path_name, '%s.zip' % local_path)
        import zipfile
        with zipfile.ZipFile('%s.zip' % local_path, 'r') as zip_ref:
            zip_ref.extractall()
    else:
        recursive = False
        if remote_model.strip()[-1] == '/':
            recursive = True
        fs.get(model_path_name, '%s' % local_path, recursive=recursive)

    print('downloaded models, at: %s' % local_path)


def download_file_to_filepath(remote_file_name, local_path=None):
    if local_path is None:
        local_path = remote_file_name
    #
    fs = get_fs()
    local_file_dir = os.path.dirname(local_path)
    if local_file_dir != '':
        os.makedirs(local_file_dir, exist_ok=True)
    if 's3://' in remote_file_name:
        remote_file_name = remote_file_name.replace('s3://', '')
    if 'aspangher' not in remote_file_name or 'edit-pathways' not in remote_file_name:
        remote_file_name = os.path.join('aspangher', 'edit-pathways', remote_file_name)
    fs.get(remote_file_name, local_path)
    print('Downloading remote filename %s -> %s' % (remote_file_name, local_path))
    return local_path
