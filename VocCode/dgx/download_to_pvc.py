import logging
import os
import zipfile
from pathlib import Path
from typing import Union

from google.cloud import storage
from tqdm import tqdm


# log = logging.getLogger(__file__)
# log.setLevel(logging.DEBUG)

bucket_namespace = "aiml-javen-research"
bucket_name = "aiml-javen-research-data"


def get_bucket(bucket_namespace: str, bucket_name: str):
    client = storage.Client(project=bucket_namespace)
    bucket = client.get_bucket(bucket_name)
    return bucket


def download_voc_unzip(data_dir: str, prefix, pvc=False):
    if pvc:
        data_dir = "/pvc/" + data_dir
    bucket_prefix = 'voc.zip'
    dst_folder = Path(data_dir)
    print('Destination Foder list ==> {}'.format(dst_folder))
    if dst_folder.exists():
        # print(os.listdir('/pvc/'))
        print('Skipping download as data dir already exists')
        return
    else:
        print('searching blob ...')
        bucket = get_bucket('aiml-javen-research',
                            'aiml-javen-research-data')
        blob = bucket.blob(prefix+bucket_prefix)
        print('downloading ...')
        path = "/pvc" if pvc else "./"
        with open(Path(path)/bucket_prefix, 'wb') as sink:
            blob.download_to_file(sink)
        print('unziping the {} ...'.format(bucket_prefix))
        with zipfile.ZipFile('{}/voc.zip'.format(path), 'r') as zip_ref:
            zip_ref.extractall('{}/'.format(path))
    print(os.listdir(path))



def upload_checkpoint(local_path: str, prefix: str, checkpoint_filepath: Union[Path, str]):
    """Upload a model checkpoint to the specified bucket in GCS."""
    bucket_prefix = prefix
    src_path = f"{local_path}/{checkpoint_filepath}"
    dst_path = f"{bucket_prefix}/{checkpoint_filepath}"
    print('Uploading {} => {}'.format(src_path, dst_path))
    bucket = get_bucket(bucket_namespace, bucket_name)
    blob = bucket.blob(dst_path)
    blob.upload_from_filename(src_path)
    print('finish uploading.')



def download_checkpoint(checkpoint_filepath: str, prefix: str, bucket_namespace='aiml-javen-research', bucket_name='aiml-javen-research-data'):
    src_path = f"yh/exercise_4/pretrained/{prefix}"
    dest_path = f"{checkpoint_filepath}/{prefix}"
    print('Downloading {} => {}'.format(src_path, checkpoint_filepath))
    bucket = get_bucket(bucket_namespace, bucket_name)
    print('searching blob ...')
    blob = bucket.blob(src_path)
    print('start downloading ...')
    
    blob.download_to_filename(dest_path)
    print('finish downloading.')

#def download_checkpoint(checkpoint_filepath: str, prefix:str, bucket_namespace: str, bucket_name: str):
#    src_path = f"{prefix}/{checkpoint_filepath}"
#    print('Downloading {} => {}'.format(checkpoint_filepath,src_path))
#    bucket = get_bucket(bucket_namespace, bucket_name)
#    blob = bucket.blob(src_path)
#    blob.download_to_filename(checkpoint_filepath)
#    print('Finish downloading')

