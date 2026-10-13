from core.auth import ROLE_RANK, generate_api_key, hash_api_key


def test_role_rank_orders_admin_above_member_above_viewer():
    assert ROLE_RANK["admin"] > ROLE_RANK["member"] > ROLE_RANK["viewer"]


def test_generate_api_key_has_recognizable_prefix_and_is_unique():
    a, b = generate_api_key(), generate_api_key()
    assert a.startswith("cxk_") and b.startswith("cxk_")
    assert a != b


def test_hash_api_key_is_deterministic_and_not_reversible_looking():
    key = generate_api_key()
    assert hash_api_key(key) == hash_api_key(key)
    assert hash_api_key(key) != key
