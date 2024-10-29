import re
from .register import Register
from .logger_config import LoggerConfig

extract_label_register = Register("extract_label")


@extract_label_register.register("default")
def default_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

    text = text.lower()
    pattern = r"final answer:\s+<begin>(.*)<end>"

    match = re.findall(pattern, text)
    if len(match) > 0:
        return match[-1]
    else:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"


@extract_label_register.register("headline_binary")
def headline_binary_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"
    text = text.lower()
    pattern = r"answer:\s+(headline 1|headline 2|other)"
    match = re.findall(pattern, text.lower())

    if match:
        answer = match[-1] if len(match) > 0 else None
        if answer == "headline 1":
            return "Headline 1 has more clicks than Headline 2."
        elif answer == "headline 2":
            return "Headline 2 has more clicks than Headline 1."
    logger.warning(f"Could not extract label from text: {text}")
    return "other"


@extract_label_register.register("hotel_reviews")
def hotel_reviews_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

    # only keep the part after "Final answer:"
    text = text.lower()

    pattern = r"final answer:\s+(truthful|deceptive|other)"

    match = re.findall(pattern, text.lower())
    if match:
        answer = match[-1] if len(match) > 0 else None
        if answer == "truthful":
            return "truthful"
        elif answer == "deceptive":
            return "deceptive"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"


@extract_label_register.register("retweet")
def retweet_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    """
    `text` follows the format "the <label> tweet got more retweets"
    """
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"
    text = text.lower()
    pattern = r"answer: the (\w+) tweet"
    match = re.findall(pattern, text)
    if len(match) > 0:
        return match[-1]
    else:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"


@extract_label_register.register("shoe")
def extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")
    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"

    pattern = r"final answer:\s+(white|red|orange|green|blue|black)"

    match = re.findall(pattern, text.lower())
    if match:
        answer = match[-1] if len(match) > 0 else None
        if answer in ["white", "red", "orange", "green", "blue", "black"]:
            return answer
    logger.warning(f"Could not extract label from text: {text}")
    return "other"

@extract_label_register.register("persuasive_pairs")
def persuasive_pairs_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")

    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"
    text = text.lower()
    patterns = [
        r"answer: the (\w+) argument",
        r"answer: \[the (\w+) argument",
        r"answer: (\w+) argument",
    ]

    prev_answer = ""
    for pattern in patterns:
        match = re.findall(pattern, text.lower())
        if match:
            answer = match[-1] if len(match) > 0 else None
            if prev_answer == "":
                prev_answer = answer
            elif prev_answer != "" and answer != prev_answer:
                return "conflict"

    for pattern in patterns:
        match = re.findall(pattern, text.lower())
        if match:
            answer = match[-1] if len(match) > 0 else None
            if answer == "first":
                return "first"
            elif answer == "second":
                return "second"
            else:
                return "other"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"

@extract_label_register.register("dreaddit")
def dreaddit_extract_label(text):
    logger = LoggerConfig.get_logger("extract_label")

    if text is None:
        logger.warning(f"Could not extract label from text: {text}")
        return "other"
    text = text.lower()
    patterns = [
        r"answer: (\w+) stress",
        r"answer: \[(\w+) stress",
    ]

    prev_answer = ""
    for pattern in patterns:
        match = re.findall(pattern, text.lower())
        if match:
            answer = match[-1] if len(match) > 0 else None
            if prev_answer == "":
                prev_answer = answer
            elif prev_answer != "" and answer != prev_answer:
                return "conflict"

    for pattern in patterns:
        match = re.findall(pattern, text.lower())
        if match:
            answer = match[-1] if len(match) > 0 else None
            if answer == "has":
                return "has stress"
            elif answer == "no":
                return "no stress"
            else:
                return "other"
    logger.warning(f"Could not extract label from text: {text}")
    return "other"