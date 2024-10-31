function a = gpuArrayZeros (varargin)
  a = ocl_zeros(varargin{:});
end
