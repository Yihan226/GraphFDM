def get_or_create(cache, key, factory):
    if key in cache:
        return cache[key]
    value = factory()
    cache[key] = value
    return value


def clear_caches(*caches):
    for cache in caches:
        cache.clear()
