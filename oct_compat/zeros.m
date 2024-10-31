function Z = zeros (varargin)
  if strcmp(varargin{end}, 'gpuArray')
    Z = gpuArrayZeros(varargin{1:end-1});
  else
    Z = builtin ("zeros", varargin{:});
  end
end
