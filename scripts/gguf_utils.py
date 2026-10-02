"""Shared GGUF metadata helpers.

The GGUF Python reader represents scalar/string fields differently across
versions.  In particular, indexing ``field.parts[field.data[0]][0]`` can return
the first byte of a string instead of the string itself.  Keep the compatibility
logic in one place so metadata consumers agree.
"""


def field_value(field):
    if field is None:
        return None
    try:
        value = field.contents()
        return value.decode("utf-8") if isinstance(value, bytes) else value
    except Exception:
        pass
    try:
        value = field.parts[field.data[0]][0]
        return value.item() if hasattr(value, "item") else value
    except Exception:
        pass
    try:
        raw = field.parts[field.data[0]]
        return bytes(raw).decode("utf-8")
    except Exception:
        return None


def read_kv(reader, key):
    return field_value(reader.fields.get(key))
