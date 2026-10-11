from __future__ import annotations

import logging
from typing import NoReturn

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    IncompleteReadError,
    ReadTimeoutError,
    ResponseStreamingError,
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

    @staticmethod
    def _raise_for_client_error(
        *,
        operation: str,
        key: str,
        exc: ClientError,
        transient_message: str,
        failure_message: str,
    ) -> NoReturn:
        """Translate a Boto3 ClientError into an application storage exception."""
        error = exc.response.get("Error", {})
        response_metadata = exc.response.get("ResponseMetadata", {})

        error_code = error.get("Code")
        status_code = response_metadata.get("HTTPStatusCode")
        request_id = response_metadata.get("RequestId")

        if error_code == "NoSuchBucket":
            logger.error(
                "S3 %s: configured bucket not found key=%s request_id=%s",
                operation,
                key,
                request_id,
                extra={
                    "operation": operation,
                    "key": key,
                    "error_code": error_code,
                    "request_id": request_id,
                },
            )
            raise StorageException(
                "Configured object storage bucket was not found"
            ) from exc

        if status_code == 404 or error_code in {"404", "NoSuchKey", "NotFound"}:
            logger.warning(
                (
                    "S3 %s: object not found "
                    "key=%s status_code=%s error_code=%s request_id=%s"
                ),
                operation,
                key,
                status_code,
                error_code,
                request_id,
                extra={
                    "operation": operation,
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
            logger.warning(
                (
                    "S3 %s: temporary failure "
                    "key=%s status_code=%s error_code=%s request_id=%s"
                ),
                operation,
                key,
                status_code,
                error_code,
                request_id,
                extra={
                    "operation": operation,
                    "key": key,
                    "status_code": status_code,
                    "error_code": error_code,
                    "request_id": request_id,
                },
            )
            raise TransientStorageException(transient_message) from exc

        logger.error(
            (
                "S3 %s: storage request failed "
                "key=%s status_code=%s error_code=%s request_id=%s"
            ),
            operation,
            key,
            status_code,
            error_code,
            request_id,
            extra={
                "operation": operation,
                "key": key,
                "status_code": status_code,
                "error_code": error_code,
                "request_id": request_id,
            },
        )
        raise StorageException(failure_message) from exc

    @staticmethod
    def _raise_for_transient_error(
        *,
        operation: str,
        key: str,
        exc: Exception,
        message: str,
    ) -> NoReturn:
        """Log and translate a transient storage failure."""

        error_type = type(exc).__name__

        logger.warning(
            "S3 %s: transient failure key=%s error_type=%s",
            operation,
            key,
            error_type,
            extra={
                "operation": operation,
                "key": key,
                "error_type": error_type,
            },
        )

        raise TransientStorageException(message) from exc

    @staticmethod
    def _raise_for_boto_core_error(
        *,
        operation: str,
        key: str,
        exc: BotoCoreError,
        message: str,
    ) -> NoReturn:
        """Log and translate an otherwise unclassified BotoCoreError."""

        error_type = type(exc).__name__

        logger.error(
            "S3 %s: storage client failure key=%s error_type=%s",
            operation,
            key,
            error_type,
            extra={
                "operation": operation,
                "key": key,
                "error_type": error_type,
            },
        )

        raise StorageException(message) from exc

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
            self._raise_for_client_error(
                operation="head_object",
                key=key,
                exc=exc,
                transient_message="Temporary storage failure while inspecting object",
                failure_message="Failed to inspect stored object metadata",
            )

        except (
            ConnectTimeoutError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        ) as exc:
            self._raise_for_transient_error(
                operation="head_object",
                key=key,
                exc=exc,
                message="Storage connection failed while inspecting object",
            )

        except BotoCoreError as exc:
            self._raise_for_boto_core_error(
                operation="head_object",
                key=key,
                exc=exc,
                message="Failed to inspect stored object metadata",
            )

    def read(self, key: str) -> bytes:
        """Read object content from S3-compatible storage."""

        body = None

        try:
            response = self._client.get_object(
                Bucket=self._bucket_name,
                Key=key,
            )
            body = response["Body"]

            return body.read()

        except ClientError as exc:
            self._raise_for_client_error(
                operation="get_object",
                key=key,
                exc=exc,
                transient_message="Temporary storage failure while reading object",
                failure_message="Failed to read stored object",
            )

        except (
            ConnectTimeoutError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
            IncompleteReadError,
            ResponseStreamingError,
        ) as exc:
            self._raise_for_transient_error(
                operation="get_object",
                key=key,
                exc=exc,
                message="Temporary storage failure while reading object",
            )

        except BotoCoreError as exc:
            self._raise_for_boto_core_error(
                operation="get_object",
                key=key,
                exc=exc,
                message="Failed to read stored object",
            )

        finally:
            if body is not None:
                try:
                    body.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        (
                            "S3 GetObject: failed to close response body "
                            "key=%s error_type=%s"
                        ),
                        key,
                        type(exc).__name__,
                        extra={
                            "operation": "get_object",
                            "key": key,
                            "error_type": type(exc).__name__,
                        },
                    )

    def save(self, key: str, content: bytes) -> None:
        """Persist content at the given storage key."""

        try:
            self._client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=content,
            )

        except ClientError as exc:
            self._raise_for_client_error(
                operation="put_object",
                key=key,
                exc=exc,
                transient_message=("Temporary storage failure while saving object"),
                failure_message="Failed to save stored object",
            )

        except (
            ConnectTimeoutError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        ) as exc:
            self._raise_for_transient_error(
                operation="put_object",
                key=key,
                exc=exc,
                message="Temporary storage failure while saving object",
            )

        except BotoCoreError as exc:
            self._raise_for_boto_core_error(
                operation="put_object",
                key=key,
                exc=exc,
                message="Failed to save stored object",
            )

    def delete(self, key: str) -> None:
        """Delete the object at the given storage key if it exists."""

        try:
            self._client.delete_object(
                Bucket=self._bucket_name,
                Key=key,
            )

        except ClientError as exc:
            self._raise_for_client_error(
                operation="delete_object",
                key=key,
                exc=exc,
                transient_message=("Temporary storage failure while deleting object"),
                failure_message="Failed to delete stored object",
            )

        except (
            ConnectTimeoutError,
            ConnectionClosedError,
            EndpointConnectionError,
            ReadTimeoutError,
        ) as exc:
            self._raise_for_transient_error(
                operation="delete_object",
                key=key,
                exc=exc,
                message="Temporary storage failure while deleting object",
            )

        except BotoCoreError as exc:
            self._raise_for_boto_core_error(
                operation="delete_object",
                key=key,
                exc=exc,
                message="Failed to delete stored object",
            )
