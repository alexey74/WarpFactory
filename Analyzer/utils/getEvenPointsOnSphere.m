function [Vector] = getEvenPointsOnSphere(R,numberOfPoints,useGPU)
%GETEVENPOINTSONSPHERE Summary of this function goes here
%   Detailed explanation goes here

if useGPU
    goldenRatio = (1+5^0.5)/2;
    Vector = gpuArrayZeros(3,numberOfPoints);
    # numberOfPoints = gpuArray(numberOfPoints);
    R = gpuArray(R);
else
    goldenRatio = (1+5^0.5)/2;
    Vector = zeros(3,numberOfPoints);
end

for i = 0:numberOfPoints-1
    theta = 2*pi*i/goldenRatio;
    phi = acos(1-2*(i+0.5)/numberOfPoints);

    Vector(1,i+1) = R*cos(theta)*sin(phi);
    Vector(2,i+1) = R*sin(theta)*sin(phi);
    Vector(3,i+1) = R*cos(phi);
end

Vector = real(Vector);
# FIXME: OclArray: octave indexing must result in a contiguous memory range
if useGPU
    Vector = gather(Vector);
endif

end


