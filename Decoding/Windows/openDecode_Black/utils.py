

def is_all_true(data:dict) -> bool:
    """check dict is all True"""
    if isinstance(data, dict):
        return all(is_all_true(value) for value in data.values())
    return data is True