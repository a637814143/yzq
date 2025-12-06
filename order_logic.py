from enum import Enum


class OrderStatus(str, Enum):
    """Represents the current status of an order on the user side."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    REFUND_REQUESTED = "refund_requested"
    CANCELLED = "cancelled"


def can_delete_order(status: OrderStatus) -> bool:
    """Determine whether a user can delete an order based on its status.

    Rules (用户端):
    - The order must be confirmed by the user, or
    - A refund must have already been requested.
    Only under these circumstances can the order be deleted from "我的订单".
    """

    return status in {OrderStatus.CONFIRMED, OrderStatus.REFUND_REQUESTED}


def can_book_timeslot(free_quota: int) -> bool:
    """Return True if a timeslot can be booked.

    When booking a service, if the available quota (空闲人数) after the time
    period is 0, the user cannot make a reservation.
    """

    return free_quota > 0
