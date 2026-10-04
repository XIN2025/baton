from app import handle

codes = [handle("alice") for _ in range(20)]
print("statuses:", codes)
print("first 429 at call", codes.index(429) + 1 if 429 in codes else "never")
