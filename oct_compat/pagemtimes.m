function Z = pagemtimes (varargin)
  transpX = "none";
  transpY = "none";
  if nargin == 2
    [X, Y] = varargin{:};
  elseif nargin == 4
    [X,transpX,Y,transpY] = varargin{:};
  endif

  if strcmp(transpX, "transpose")
    X = pagetranspose(X) ;
  end
  if strcmp(transpY, "transpose")
    Y = pagetranspose(Y) ;
  end

  Z = multiprod_legacy(X, Y);

endfunction

%!test "multiply 3d arrays"
%!
%! X(:,:,1) = [
%!    5     1
%!    6     6
%! ];
%! X(:,:,2) = [
%!  4     2
%!  1     4
%! ];
%! X(:,:,3) = [
%!  6     1
%!  6     6
%! ];
%! Y(:,:,1) = [
%!     6     5
%!     3     1
%! ];
%! Y(:,:,2) = [
%!     3     5
%!     6     6
%! ];
%! Y(:,:,3) = [
%!     4     6
%!     1     6
%! ];
%! Z = pagemtimes(X,Y);
%! assert(size(Z) == [2 2 3])
%! assert (Z(:,:,1) == [ 33    26
%!    54    36
%! ]);
%!
%!
%!test "multiply 3d arrays with transpose"
%!
%! X(:,:,1) = [
%!    5     1
%!    6     6
%! ];
%! X(:,:,2) = [
%!  4     2
%!  1     4
%! ];
%! X(:,:,3) = [
%!  6     1
%!  6     6
%! ];
%! Y(:,:,1) = [
%!     6     5
%!     3     1
%! ];
%! Y(:,:,2) = [
%!     3     5
%!     6     6
%! ];
%! Y(:,:,3) = [
%!     4     6
%!     1     6
%! ];
%! Z = pagemtimes(X,"transpose",Y,"none");
%! assert (Z(:,:,1) == [    48   31
%!   24   11
%! ]);
%! assert(size(Z) == [ 2 2 3 ]) ;
%!
%!test "Multiply 6-D Arrays"
%! X = ones(3,3,2,2,2,2);
%! A = eye(3);
%! Y = cat(5,A,2*A,3*A,4*A,5*A,6*A);
%! Z = pagemtimes(X,Y)
%! assert (size(Z) == [3 3 3 3 3 3])
%! assert( Z(:,:,1,3,4,5) == [ 5 5 5 5 ; 5 5 5 5 ; 5 5 5 5 ] )
%!