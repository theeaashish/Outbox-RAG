from __future__ import annotations

import logging

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from app.core.exceptions import (
    ResourceNotFoundException,
    StorageException,
    TransientStorageException,
)
from app.core.storage.base import StorageService, StoredObjectMetadata
from app.core.storage.presigned import PresignedUrlService

logger = logging.getLogger(__name__)


class S3ObjectStorage(StorageService, PresignedUrlService):
    """S3-compatible object storage adapter."""

    def __init__(
        self,
        *,
        bucket_name: str,
        endpoint_url: str,
        region_name: str,
        access_key_id: str,
        secret_access_key: str,
    ) -> None:
        self._bucket_name = bucket_name

        self._client: BaseClient = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                signature_version="s3v4",
                connect_timeout=5,
                read_timeout=30,
                retries={
                    "mode": "standard",
                    "max_attempts": 3,
                },
                s3={
                    "addressing_style": "virtual",
                },
            ),
        )

    def head(self, key: str) -> StoredObjectMetadata:
        """Return metadata for an object without downloading its contents."""

        try:
            response = self._client.head_object(
                Bucket=self._bucket_name,
                Key=key,
            )

            return StoredObjectMetadata(
                size=response["ContentLength"],
                content_type=response.get("ContentType"),
                etag=response.get("ETag"),
            )

        except ClientError as exc:
            error = exc.response.get("Error", {})
            response_metadata = exc.response.get("ResponseMetadata", {})

            error_code = error.get("Code")
            status_code = response_metadata.get("HTTPStatusCode")

            if status_code == 404 or error_code in {"404", "NoSuchKey", "NotFound"}:
                request_id = response_metadata.get("RequestId")

                logger.warning(
                    (
                        "S3 HeadObject: object not found "
                        "key=%s status_code=%s error_code=%s request_id=%s"
                    ),
                    key,
                    status_code,
                    error_code,
                    request_id,
                    extra={
                        "operation": "head_object",
                        "key": key,
                        "status_code": status_code,
                        "error_code": error_code,
                        "request_id": request_id,
                    },
                )
                raise ResourceNotFoundException("Stored object not found") from exc

            if status_code in {408, 429, 500, 502, 503, 504} or error_code in {
                "SlowDown",
                "RequestTimeout",
                "InternalError",
                "ServiceUnavailable",
                "Throttling",
                "ThrottlingException",
            }:
                raise TransientStorageException(
                    "Temporary storage failure while inspecting object"
                ) from exc

            raise StorageException("Failed to inspect stored object metadata") from exc

        except (
            ConnectTimeoutError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        ) as exc:
            raise TransientStorageException(
                "Storage connection failed while inspecting object"
            ) from exc

        except BotoCoreError as exc:
            raise StorageException("Failed to inspect stored object metadata") from exc
