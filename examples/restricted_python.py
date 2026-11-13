# Run: wisp examples/restricted_python.py (no external Python installation)
# Computation uses Wisp's Rust Tensor implementation, not NumPy or CPython.
a = tensor([1, 2, 3, 4], 2, 2)


def twice(x):
    return x + x


b = twice(a)
c = a @ b
assert c.shape() == (2, 2)
assert c.values() == [14.0, 20.0, 30.0, 44.0]
assert a.T.values() == [1.0, 3.0, 2.0, 4.0]
print(c.values())
