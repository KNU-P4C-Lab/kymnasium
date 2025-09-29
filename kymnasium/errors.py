class InvalidActionError(Exception):
    def __init__(self, action):
        self._action = action

    def __str__(self):
        return f"Invalid action: {self._action}"


class NotAllowedUserIdError(Exception):
    def __init__(self, user_id):
        self._user_id = user_id

    def __str__(self):
        return f"Not allowed user id: {self._user_id}"


class ObservationError(Exception):
    def __init__(self, message: str):
        self._message = message

    def __str__(self):
        return self._message

class EnvironmentClosedError(Exception):
    def __init__(self, message: str):
        self._message = message

    def __str__(self):
        return self._message