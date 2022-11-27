function [uOpt]=qp_supervisor(ABarrier,bBarrier,uRef,varargin)
method='quadprog';

%optional parameters
ivarargin=1;
while(ivarargin<=length(varargin))
    switch(lower(varargin{ivarargin}))
        case 'method'
            ivarargin=ivarargin+1;
            method=lower(varargin{ivarargin});
        otherwise
            disp(varargin{ivarargin})
            error('Argument not valid!')
    end
    ivarargin=ivarargin+1;
end

switch method
    case 'cvx'
        cvx_begin quiet
            variables u(2,1)
            uDiff=u-uRef;
            minimize(uDiff'*uDiff)
            %Octave version: minimize pow_pos(norm(u-uRef,2),2)
            subject to
            ABarrier*u+bBarrier<=0
        cvx_end
        if ~strcmp(cvx_status,'Solved')
            %The problem is infeasible
            error('The QP problem was not solved because CVX found it %s',cvx_status)
        end
    case 'quadprog'
        H=eye(size(uRef,1));
        f=-uRef;
        A=ABarrier;
        b=-bBarrier; %quadprog uses the constraint Ax<=b instead of Ax+b<=0
        opts=optimset('Display','None');
        [u,~,exitFlag]=quadprog(H,f,A,b,[],[],[],[],uRef,opts);
        if exitFlag<=0
            %There was a problem in the optimization
            error('The QP problem was not solved (exitFlag = %d)', exitFlag)
        end
        flagFeasible=all(ABarrier*u<=b+1e-9);
        if ~flagFeasible
            %The problem is infeasible
            error('quadprog did not find a feasible solution (exitFlag = %d)',exitFlag)
        end
end
uOpt=u;
