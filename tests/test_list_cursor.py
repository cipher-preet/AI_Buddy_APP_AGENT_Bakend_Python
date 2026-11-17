from datetime import datetime, timezone

from bson import ObjectId

from services.db.list_cursor import cursor_match, decode_list_cursor, encode_list_cursor


def test_list_cursor_round_trip():
    stamp = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    doc_id = ObjectId()
    encoded = encode_list_cursor(stamp, 3, doc_id)
    decoded_stamp, version, decoded_id = decode_list_cursor(encoded)
    assert decoded_stamp == stamp
    assert version == 3
    assert decoded_id == doc_id


def test_list_cursor_rejects_garbage():
    try:
        decode_list_cursor("not-a-cursor")
    except ValueError as error:
        assert "Invalid list cursor" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_cursor_match_follows_sort():
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    doc_id = ObjectId()
    by_time = cursor_match(stamp, 2, doc_id, version_first=False)
    assert "$or" in by_time
    assert by_time["$or"][0] == {"updatedAt": {"$lt": stamp}}
    by_version = cursor_match(stamp, 2, doc_id, version_first=True)
    assert by_version["$or"][0] == {"version": {"$lt": 2}}
