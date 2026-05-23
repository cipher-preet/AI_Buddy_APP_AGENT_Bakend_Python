def success_response(data=None, message="Success"):
    return {
        "success": True,
        "message": message,
        "data": data
    }


def error_response(message="Something went wrong"):
    return {
        "success": False,
        "message": message
    }