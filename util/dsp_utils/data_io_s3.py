import io
import logging
import os


def connect_to_s3(access_key=None, key_id=None, url=None, http_proxy=""):
    """
    Create a connection to S3 with a boto3 session using the specified
    credentials.

    If a credential is not provided (such as access_key, key_id, and url),
    the environment variables are searched for the appropriate values.

    Args:
        (optional) access_key: The AWS secret access key for access to
                               ml-patrec S3.
                               Default value: the value of environment variable
                               AWS_SECRET_ACCESS_KEY
        (optional) key_id: The AWS access key ID for ml-patrec S3
                           Default value: the value of environment variable
                           AWS_ACCESS_KEY_ID
        (optional) url: The url of the AWS endpoint
                        Default value: the value of environment variable
                        AWS_ENDPOINT
    Returns:
        An S3 connection defined by a resource from a boto3 session
    """

    # Get default values from enivronment if necessary
    if access_key is None:
        access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    if key_id is None:
        key_id = os.environ["AWS_ACCESS_KEY_ID"]
    if url is None:
        url = os.environ["AWS_ENDPOINT"]

    # Move imports here to surpress logs
    import boto3
    import botocore
    import s3transfer # NOQA

    # Disable Excessive boto3 log
    for name in logging.Logger.manager.loggerDict.keys():
        if ('boto' in name) or ('urllib3' in name) or ('s3transfer' in name) \
           or ('boto3' in name) or ('botocore' in name) or ('nose' in name):
            logging.getLogger(name).setLevel(logging.INFO)

    session = boto3.session.Session()
    s3 = session.resource(
        "s3",
        endpoint_url=url,
        aws_access_key_id=key_id,
        aws_secret_access_key=access_key,
        config=botocore.config.Config(
            signature_version="s3v4",
            retries={"max_attempts": 3},
            proxies={"http": http_proxy}
        ),
    )

    return s3


class S3File(io.IOBase):
    """
    A file handle that reads and writes to S3 using the specified S3 connection
    and bucket name.
    """
    def __init__(self, s3_connection, s3_bucket_name, s3_file_name, mode):
        """
        Args:
            s3_connection: A boto3 resource sepecifying the AWS S3 connection
            s3_bucket_name: The name of the S3 bucket to access
            s3_file_name: The name of the file within the S3 bucket to access
            mode: The mode to read or write from the file. The supported
                  options are:
                    "rb": Read binary
                    "rt": Read text
                    "wb": Write binary
                    "wt": Write text
        """

        s3_bucket = s3_connection.Bucket(s3_bucket_name)
        self.s3_object = s3_bucket.Object(s3_file_name)
        self.mode = mode

        try:

            # READ
            if "rb" in mode:
                self.buffer = io.BytesIO()
                self.s3_object.download_fileobj(self.buffer)
                self.buffer.seek(0)
            elif "rt" in mode:
                binary_buffer = io.BytesIO()
                self.s3_object.download_fileobj(binary_buffer)
                self.buffer = io.StringIO(
                    binary_buffer.getvalue().decode("UTF-8"))

            # WRITE
            else:
                self.buffer = None

                if "wb" not in mode and "wt" not in mode:
                    raise NotImplementedError("Unknown mode: {}".format(mode))

        except Exception as exc:
            raise RuntimeError(
                "Fail to read '{}' from bucket '{}' !".format(
                    s3_file_name,
                    s3_bucket_name)) from exc

    def check_read_permissions(self):
        """
        Checks if this object has read perissions.
        If not, a PermissionError is raised
        """
        if "wb" in self.mode or "wt" in self.mode:
            raise PermissionError(
                "Method not supported for write mode `{}`".format(self.mode))

    def check_write_permissions(self):
        """
        Checks if this object has write perissions.
        If not, a PermissionError is raised
        """
        if "rb" in self.mode or "rt" in self.mode:
            raise PermissionError(
                "Method not supported for write mode `{}`".format(self.mode))

    def read(self, length=-1):
        """
        Returns: The next item in the read buffer
        """
        self.check_read_permissions()
        return self.buffer.read(length)

    def readlines(self):
        """
        Returns: All lines in the read buffer
        """
        self.check_read_permissions()
        return self.buffer.readlines()

    def write(self, data):
        """
        Args:
            data: The information to write to S3
        """
        self.check_write_permissions()

        if "wt" in self.mode:
            data = data.encode()

        buf = io.BytesIO(data)

        return self.s3_object.upload_fileobj(buf)
