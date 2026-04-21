def parse_file(file):
    return {
        "functions": ["login", "checkout", "payment"],
        "calls": [("login", "checkout"), ("checkout", "payment")]
    }