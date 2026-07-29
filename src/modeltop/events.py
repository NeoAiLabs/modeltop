"""Application-internal coalescible state notifications."""

from textual.message import Message


class DashboardStateChanged(Message, bubble=False):
    """Signal that widgets should read the latest shared state snapshot."""

    def can_replace(self, message: Message) -> bool:
        return isinstance(message, DashboardStateChanged)
