from app.tools import verification as v


def test_official_demo_fields_verify_margaret():
    r = v.evaluate({"full_name": "Margaret Chen", "policy_number": "POL-9921",
                    "dob": "1985-03-15", "id_last4": "4472"})
    assert r.verified and r.party_id == "P9"
    assert sorted(r.matched) == ["dob", "full_name", "id_last4"]


def test_policy_number_does_not_count_toward_threshold():
    r = v.evaluate({"policy_number": "POL-9921", "full_name": "Margaret Chen", "dob": "1985-03-15"})
    assert not r.verified and not r.ready


def test_fewer_than_three_fields_is_not_ready():
    r = v.evaluate({"full_name": "Margaret Chen", "dob": "1985-03-15"})
    assert not r.ready and not r.failed


def test_any_mismatch_fails_once_ready():
    r = v.evaluate({"full_name": "Margaret Chen", "dob": "1985-03-15", "id_last4": "0000"})
    assert r.failed and r.mismatched == ["id_last4"]


def test_unknown_policy_number_is_a_mismatch():
    r = v.evaluate({"policy_number": "POL-0000", "full_name": "Margaret Chen",
                    "dob": "1985-03-15", "id_last4": "4472"})
    assert r.failed and "policy_number" in r.mismatched


def test_name_alias_and_email_alias():
    r = v.evaluate({"full_name": "Yaven Li", "email": "YAWEN.LI@example.com", "dob": "December 3, 1989"})
    assert r.verified and r.party_id == "P13"


def test_national_id_last4_and_phone_formats():
    r = v.evaluate({"full_name": "Ma Tian", "id_last4": "6688", "phone": "(650) 208-8799"})
    assert r.verified and r.party_id == "P12"


def test_name_order_and_dob_formats():
    assert v.field_matches({"name": "Margaret Chen"}, "full_name", "chen margaret")
    assert v.normalize_dob("03/15/1985") == "1985-03-15"
    assert v.normalize_dob("March 15th, 1985") == "1985-03-15"


def test_representative_authorization():
    assert v.is_authorized_representative("David Chen", "P9")
    assert not v.is_authorized_representative("David Chen", "P7")
    assert not v.is_authorized_representative("John Smith", "P9")
