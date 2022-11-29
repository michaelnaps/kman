%function sphere_plot(sphere,color)
%This function draws the sphere (i.e., a circle) with radius  sphere.r and the
%specified color, and another circle with radius  sphere.rInfluence in gray.
function varargout = plot_sphere(sphere,color)
TOL = 1e-6;
%geometrical radius
radius=abs(sphere.radius);
%filled-in or hollow
radiusSign=sign(sphere.radius);
if radiusSign>0
    faceColor=[0.8 0.8 0.8];
else
    faceColor=[1 1 1];
end
%geometry radius of influence
radiusInfluence=radius+radiusSign*sphere.distInfluence;
%plotting of the sphere and sphere of influence
plotHandle1=plotCircle(sphere.xCenter(1),sphere.xCenter(2),radius,'EdgeColor',color,'FaceColor',faceColor,'linewidth',1.5);

if radiusInfluence > TOL
    plotHandle2=plotCircle(sphere.xCenter(1),sphere.xCenter(2),radiusInfluence,'EdgeColor',[0.5 0.5 0.5]);
end

if nargout>0
    varargout{1}=[plotHandle1;plotHandle2];
end


function plotHandle=plotCircle(xCenter,yCenter,radius,varargin)
diameter = radius*2;
xCorner = xCenter-radius;
yCorner = yCenter-radius;
plotHandle = rectangle('Position',[xCorner yCorner diameter diameter],'Curvature',[1,1],varargin{:});
