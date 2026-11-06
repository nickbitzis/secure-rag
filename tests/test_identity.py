from securerag.identity import sign, verify


def test_sign_and_verify():
    assert verify(sign("hal-ben", "s3"), "s3") == "hal-ben"


def test_tampered_user_id_rejected():
    tok = sign("hal-frank", "s3")
    forged = "hal-erin." + tok.split(".", 1)[1]
    assert verify(forged, "s3") is None


def test_wrong_secret_and_garbage_rejected():
    assert verify(sign("hal-ben", "a"), "b") is None
    assert verify("no-dot-here", "a") is None
    assert verify("", "a") is None
