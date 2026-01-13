def extract_attributes(text):
    text = text.lower()

    attrs = {
        "gender": "male" if "male" in text or "man" in text else "female",
        "face_shape": "oval" if "oval" in text else "round",
        "hair_length": "short" if "short" in text else "long",
        "hair_color": "black" if "black" in text else "brown",
        "beard": "yes" if "beard" in text else "no",
        "glasses": "yes" if "glasses" in text else "no"
    }

    return attrs
