def fetch_data():
    raw = load_raw()
    return raw

def load_raw():
    return [1, 2, 3]

def process(data):
    cleaned = clean(data)
    result = transform(cleaned)
    return result

def clean(data):
    return [x for x in data if x is not None]

def transform(data):
    return [x * 2 for x in data]

def run():
    data = fetch_data()
    output = process(data)
    return output