import pytest

from order_logic import OrderStatus, can_book_timeslot, can_delete_order


@pytest.mark.parametrize(
    "status, expected",
    [
        (OrderStatus.PENDING, False),
        (OrderStatus.CONFIRMED, True),
        (OrderStatus.REFUND_REQUESTED, True),
        (OrderStatus.CANCELLED, False),
    ],
)
def test_can_delete_order(status, expected):
    assert can_delete_order(status) is expected


@pytest.mark.parametrize("free_quota, expected", [(1, True), (5, True), (0, False), (-1, False)])
def test_can_book_timeslot(free_quota, expected):
    assert can_book_timeslot(free_quota) is expected
