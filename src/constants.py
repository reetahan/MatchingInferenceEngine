
DISTRICT_TO_BOROUGH_MAPPING = {
        str(d): b for d, b in
        [(d,'M') for d in range(1,7)] +
        [(d,'X') for d in range(7,13)] +
        [(d,'K') for d in range(13,24)] +
        [(d,'Q') for d in range(24,31)] +
        [(31,'R'), (32,'K')]
    }