# Trusted wrappers compiled into Wisp. All state and policy enforcement is Rust-owned.
class Tensor:
    def __init__(self, handle):
        self.handle = handle

    def shape(self):
        return wisp_shape(self.handle)

    def values(self):
        return wisp_values(self.handle)

    def transpose(self):
        return Tensor(wisp_transpose(self.handle))

    @property
    def T(self):
        return self.transpose()

    def matmul(self, other):
        return Tensor(wisp_matmul(self.handle, other.handle))

    def add(self, other):
        return Tensor(wisp_add(self.handle, other.handle))

    def __matmul__(self, other):
        return self.matmul(other)

    def __add__(self, other):
        return self.add(other)


def tensor(data, rows, cols):
    return Tensor(wisp_tensor(data, rows, cols))


def zeros(rows, cols):
    return Tensor(wisp_zeros(rows, cols))
