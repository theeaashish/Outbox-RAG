from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock


class ChatStreamState(StrEnum):
    """Logical lifecycle states owned by the streaming runtime."""

    STREAMING = "streaming"
    CANCELLED = "cancelled"
    PROVIDER_COMPLETE = "provider_complete"
    VALIDATING = "validating"
    FINALIZING = "finalizing"
    ASSISTANT_DURABLE = "assistant_durable"
    DONE = "done"
    FAILED = "failed"


class ChatStreamCancelReason(StrEnum):
    """Reasons that can cancel an active streaming generation."""

    CLIENT_DISCONNECT = "client_disconnect"
    TTFT_TIMEOUT = "ttft_timeout"
    IDLE_TIMEOUT = "idle_timeout"
    TOTAL_TIMEOUT = "total_timeout"
    OUTPUT_LIMIT = "output_limit"


@dataclass(frozen=True, slots=True)
class ChatStreamSnapshot:
    """Immutable view of the current streaming lifecycle."""

    state: ChatStreamState
    cancel_reason: ChatStreamCancelReason | None


class ChatStreamLifecycle:
    """
    Thread-safe lifecycle coordinator for one streaming chat turn.

    The controller and the synchronous provider worker may access this object
    concurrently. The lock establishes the linearization point between
    cancellation and provider completion.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._state = ChatStreamState.STREAMING
        self._cancel_reason: ChatStreamCancelReason | None = None

    @property
    def snapshot(self) -> ChatStreamSnapshot:
        """Return an immutable snapshot of the current lifecycle state."""

        with self._lock:
            return ChatStreamSnapshot(
                state=self._state,
                cancel_reason=self._cancel_reason,
            )

    def accepts_provider_output(self) -> bool:
        """
        Return whether provider deltas may still be accepted.

        This is intentionally a read of the protected lifecycle state rather
        than a promise that the subsequent network write will succeed.
        """

        with self._lock:
            return self._state is ChatStreamState.STREAMING

    def request_cancel(self, reason: ChatStreamCancelReason) -> bool:
        """
        Attempt to cancel an active stream.

        Returns True only when this call wins the cancellation race while the
        lifecycle is still STREAMING.
        """

        with self._lock:
            if self._state is not ChatStreamState.STREAMING:
                return False

            self._state = ChatStreamState.CANCELLED
            self._cancel_reason = reason
            return True

    def accept_provider_completion(self) -> bool:
        """
        Atomically claim provider completion.

        Returns True only if the provider completion wins the race against
        cancellation.
        """

        with self._lock:
            if self._state is not ChatStreamState.STREAMING:
                return False

            self._state = ChatStreamState.PROVIDER_COMPLETE
            return True

    def begin_validation(self) -> None:
        """Transition from accepted provider completion into validation."""

        self._transition(
            expected=ChatStreamState.PROVIDER_COMPLETE,
            target=ChatStreamState.VALIDATING,
        )

    def begin_finalization(self) -> None:
        """Transition from successful validation into persistence."""

        self._transition(
            expected=ChatStreamState.VALIDATING,
            target=ChatStreamState.FINALIZING,
        )

    def mark_assistant_durable(self) -> None:
        """Record that the assistant message has been committed."""

        self._transition(
            expected=ChatStreamState.FINALIZING,
            target=ChatStreamState.ASSISTANT_DURABLE,
        )

    def mark_done(self) -> None:
        """Record that the complete application lifecycle has settled."""

        self._transition(
            expected=ChatStreamState.ASSISTANT_DURABLE,
            target=ChatStreamState.DONE,
        )

    def mark_failed(self) -> bool:
        """
        Transition the lifecycle to FAILED unless it is already terminal.

        Returns False when cancellation or another terminal state already won.
        """

        with self._lock:
            if self._state in {
                ChatStreamState.CANCELLED,
                ChatStreamState.FAILED,
                ChatStreamState.ASSISTANT_DURABLE,
                ChatStreamState.DONE,
            }:
                return False

            self._state = ChatStreamState.FAILED
            return True

    def _transition(
        self,
        *,
        expected: ChatStreamState,
        target: ChatStreamState,
    ) -> None:
        with self._lock:
            if self._state is not expected:
                raise RuntimeError(
                    f"Invalid chat stream transition: "
                    f"{self._state.value} -> {target.value}; "
                    f"expected current state {expected.value}"
                )

            self._state = target
