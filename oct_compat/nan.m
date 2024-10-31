function retval = nan (varargin)
  if strcmp(varargin{end}, 'gpuArray')
    retval = gpuArray(nan(varargin{1:end-1}));
  else
    retval = builtin ("nan", varargin{:});
  end
end
