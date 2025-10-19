def error_handle(function):
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except object as e:
            return {"error": str(e)}
    return wrapped