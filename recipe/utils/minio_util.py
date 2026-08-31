import os
import shutil
from minio import Minio
from minio.error import S3Error
import logging
import sys

logger = logging.getLogger(__name__)
# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)


class MinioManager:
    """
    A class to manage file operations for a specific, pre-existing MinIO bucket.

    This class connects to a MinIO server and is bound to a single bucket.
    If the specified bucket does not exist, an exception will be raised.
    """

    def __init__(self, bucket_name, endpoint, access_key, secret_key, secure=False):
        """
        Initializes the MinioManager and verifies the bucket's existence.

        :param bucket_name: The name of the bucket this instance will manage.
        :param endpoint: The URL or IP address and port of the MinIO server.
        :param access_key: The access key for authentication.
        :param secret_key: The secret key for authentication.
        :param secure: A boolean flag to enable or disable TLS/SSL (HTTPS).
        :raises ValueError: If the specified bucket does not exist on the server.
        """
        self.bucket_name = bucket_name
        self.client = None

        try:
            # 1. Initialize MinIO client
            self.client = Minio(
                endpoint, access_key=access_key, secret_key=secret_key, secure=secure
            )
            # 2. Check if the bucket exists. If not, throw an exception.
            found = self.client.bucket_exists(self.bucket_name)
            if not found:
                raise ValueError(
                    f"Bucket '{self.bucket_name}' does not exist. Please create it first."
                )

        except S3Error as exc:
            logger.error(f"An S3 error occurred during initialization: {exc}")
            raise
        except Exception as exc:
            logger.error(f"An error occurred during initialization: {exc}")
            raise

    def upload_file(self, object_name, file_path):
        """
        Uploads a single file to the instance's bucket.

        :param object_name: The desired name for the object in the bucket.
        :param file_path: The local path of the file to upload.
        :return: True if upload was successful, False otherwise.
        """
        if not self.client:
            logger.error("Cannot upload: MinIO client is not initialized.")
            return False

        if not os.path.exists(file_path):
            logger.error(f"Error: Local file not found at '{file_path}'")
            return False

        try:
            logger.info(
                f"Uploading '{file_path}' as '{object_name}' to bucket '{self.bucket_name}'..."
            )
            result = self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path,
            )
            logger.info(
                f"Successfully uploaded: etag={result.etag}, version_id={result.version_id}"
            )
            return True

        except S3Error as exc:
            logger.error(f"An S3 error occurred during upload: {exc}")
            raise
        except Exception as exc:
            logger.error(f"An unexpected error occurred during upload: {exc}")
            raise

    def download_file(self, object_name, file_path):
        """
        Downloads an object from the instance's bucket to a local file.

        :param object_name: The name of the object to download.
        :param file_path: The local path where the file will be saved.
        :return: True if download was successful, False otherwise.
        """
        if not self.client:
            logger.error("Cannot download: MinIO client is not initialized.")
            return False

        try:
            logger.info(
                f"Downloading object '{object_name}' from bucket '{self.bucket_name}' to '{file_path}'..."
            )
            self.client.fget_object(self.bucket_name, object_name, file_path)
            logger.info("Successfully downloaded object.")
            return True

        except S3Error as exc:
            if exc.code == "NoSuchKey":
                logger.error(
                    f"Error: Object '{object_name}' not found in bucket '{self.bucket_name}'."
                )
            else:
                logger.error(f"An S3 error occurred during download: {exc}")
        except Exception as exc:
            logger.error(f"An unexpected error occurred during download: {exc}")
            raise

        return False

    def upload_dir(self, local_dir_path: str, remote_dir_path: str):
        """
        Recursively uploads the contents of a local directory to a specified
        path within the MinIO bucket.

        :param local_dir_path: The path to the local directory to upload.
        :param remote_dir_path: The destination "folder" in the MinIO bucket.
        :return: True if all files were uploaded successfully, False otherwise.
        """
        if not os.path.isdir(local_dir_path):
            logger.error(f"Error: Provided path '{local_dir_path}' is not a directory.")
            return False

        logger.info(
            f"--- Starting directory upload from '{local_dir_path}' to '{remote_dir_path}' ---"
        )

        all_successful = True

        for root, _, files in os.walk(local_dir_path):
            for filename in files:
                local_file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_file_path, local_dir_path)
                remote_object_name = os.path.join(
                    remote_dir_path, relative_path
                ).replace("\\", "/")

                if not self.upload_file(remote_object_name, local_file_path):
                    all_successful = False
                    logger.error(
                        f"Failed to upload {local_file_path}. Continuing with other files."
                    )

        if all_successful:
            logger.info(
                f"--- Successfully uploaded all files from '{local_dir_path}' ---"
            )
        else:
            logger.error(
                f"--- Directory upload from '{local_dir_path}' completed with one or more failures. ---"
            )

        return all_successful

    def download_dir(self, remote_dir_path: str, local_dir_path: str):
        """
        Recursively downloads the contents of a "folder" from MinIO to a local directory.

        :param remote_dir_path: The source "folder" in the MinIO bucket.
        :param local_dir_path: The path to the local directory to download into.
        :return: True if all objects were downloaded successfully, False otherwise.
        """
        logger.info(
            f"--- Starting directory download from '{remote_dir_path}' to '{local_dir_path}' ---"
        )

        all_successful = True

        # Ensure the remote_dir_path ends with a slash to properly list objects in that "folder"
        if not remote_dir_path.endswith("/"):
            remote_dir_path += "/"

        objects = self.client.list_objects(
            self.bucket_name, prefix=remote_dir_path, recursive=True
        )

        for obj in objects:
            # Construct the full local path for the object
            relative_path = os.path.relpath(obj.object_name, remote_dir_path)
            local_file_path = os.path.join(local_dir_path, relative_path)

            # Download the object
            if not self.download_file(obj.object_name, local_file_path):
                all_successful = False
                logger.error(
                    f"Failed to download {obj.object_name}. Continuing with other objects."
                )

        if all_successful:
            logger.info(
                f"--- Successfully downloaded all objects from '{remote_dir_path}' ---"
            )
        else:
            logger.error(
                f"--- Directory download from '{remote_dir_path}' completed with one or more failures. ---"
            )

        return all_successful


if __name__ == "__main__":
    """
    Main function to demonstrate the MinioManager class.
    """
    # --- Configuration ---
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "45.82.79.11:19000")
    minio_access_key = os.getenv("MINIO_ACCESS_KEY", "kdirQYGuPUTkTTQaI0uF")
    minio_secret_key = os.getenv(
        "MINIO_SECRET_KEY", "3SsnbNdbZHLLl291uZikqeAN9zNFjzZF03bnqgYa"
    )
    minio_secure = "true" in os.getenv("MINIO_SECURE", "false").lower()
    target_bucket = "mlflow-artifacts"

    path = "2/b937524913c54685b831764efed00a02/artifacts"

    try:
        # 1. Instantiate the manager with a specific bucket name.
        # This will fail if the bucket doesn't exist.
        manager = MinioManager(
            bucket_name=target_bucket,
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=minio_secure,
        )
        demo_remote_path = "demo/run_xyz/artifacts"

        # --- UPLOAD DEMO ---
        logger.info("\n--- Starting Upload Demo ---")
        upload_file_path = "local-upload-file.txt"
        with open(upload_file_path, "w") as f:
            f.write(
                f"This file was uploaded at {__file__} on {__import__('datetime').datetime.now()}.\n"
            )

        # Note: bucket name is no longer passed to the method
        manager.upload_file(f"{path}/my-test-file.txt", upload_file_path)

        # --- DOWNLOAD DEMO ---
        logger.info("\n--- Starting Download Demo ---")
        download_file_path = "local-downloaded-file.txt"

        # Note: bucket name is no longer passed to the method
        if manager.download_file(f"{path}/my-test-file.txt", download_file_path):
            logger.info("\nDownload verified.")

        # --- UPLOAD DIR DEMO ---
        logger.info("\n\n--- Starting Directory Upload Demo ---")
        local_upload_dir = "local_upload_dir_test"
        sub_dir = os.path.join(local_upload_dir, "models", "sklearn")
        os.makedirs(sub_dir, exist_ok=True)

        with open(os.path.join(local_upload_dir, "requirements.txt"), "w") as f:
            f.write("scikit-learn==1.2.2\n")
        with open(os.path.join(sub_dir, "model.pkl"), "w") as f:
            f.write("dummy model data\n")

        remote_dest_dir = os.path.join(demo_remote_path, "uploaded_directory")
        manager.upload_dir(local_upload_dir, remote_dest_dir)

        # --- DOWNLOAD DIR DEMO ---
        logger.info("\n\n--- Starting Directory Download Demo ---")
        local_download_dir = "local_download_dir_test"
        manager.download_dir(remote_dest_dir, local_download_dir)

        logger.info("\n--- Verifying downloaded directory contents ---")
        for root, _, files in os.walk(local_download_dir):
            for name in files:
                logger.info(f"Found downloaded file: {os.path.join(root, name)}")

        # --- Cleanup ---
        logger.info("\n\n--- Cleaning up local files and directories ---")
        if os.path.isdir(local_upload_dir):
            shutil.rmtree(local_upload_dir)
            logger.info(f"Removed directory: '{local_upload_dir}'")
        if os.path.isdir(local_download_dir):
            shutil.rmtree(local_download_dir)
            logger.info(f"Removed directory: '{local_download_dir}'")

        # --- Cleanup ---
        logger.info("\n--- Cleaning up local files ---")
        for f in [upload_file_path, download_file_path]:
            if os.path.exists(f):
                os.remove(f)
                logger.info(f"Removed '{f}'")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except ConnectionError as e:
        logger.error(
            f"Could not connect to MinIO. Please check your credentials and endpoint. Error: {e}"
        )
    except Exception as e:
        logger.error(f"A general error occurred in main: {e}")
