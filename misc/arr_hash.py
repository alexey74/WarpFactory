# Source: https://stackoverflow.com/questions/66354598/deterministic-method-to-hash-np-array-int
"""
To test some functions a check, if a Tensor is correct, is required.
Due to the sheer size of an even ordinary 6D Tensor for certain metrics, it is easiest to have a unique identifier for such.
Since I haven't yet come up with anything better, that is defined as unique on a Tensor, the Hash Value is going to be used.
"""
import hashlib


def arr_hash(arr):
  hashed = hashlib.sha3_512(arr.tobytes())
  for dim in arr.shape:
    hashed.update(dim.to_bytes(4, byteorder='big'))
  return hashed.hexdigest()