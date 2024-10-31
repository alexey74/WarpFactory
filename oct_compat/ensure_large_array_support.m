% from https://wiki.octave.org/Enable_large_arrays:_Build_octave_such_that_it_can_use_arrays_larger_than_2Gb.
function c = ensure_large_array_support ()
  clear all;
  N = 2^31;
  ## The following line requires about 8 GB of RAM!
  a = b = ones (N, 1, "double");
  c = a' * b;
end