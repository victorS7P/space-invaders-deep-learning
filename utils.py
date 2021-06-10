import numpy as np

def to_int (bin):
  return "{x}".format(
    x=sum(1<<i for i, b in enumerate(bin) if b)
  )

def to_bin (i, s = 9):
  a = np.array([int(x) for x in bin(int(i))[2:]])
  a = np.array(np.flip(a))
  a.resize(s)

  return a