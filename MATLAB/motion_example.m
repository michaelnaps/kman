function motion_example()

% Setup the figure window
f = figure;
set(f,'KeyPressFcn', {@figure_KeyPressFcn});    % Set up the callback
set(f,'UserData', [0 0 0 0]);                   % store the x-position, y-position, x-velocity, y-velocity

userdata = get(f,'UserData');                   % retrieve the stored motion data
velocity = userdata(1:2);                       % extract motion data to variables
position = userdata(3:4);                       % extract motion data to variables

plot(position(1), position(2),'o');             % initial plot
axis([-100 100 -100 100]);

%setup the timer to repeat an infinite number of times at a 0.1 sec period
% specify the move_ball function to execute preriodically
% with an input parameter of the figure f - needed so that we can get the
% motion data when inside the callback
t = timer('TimerFcn', {@move_ball, f}, 'Period', 0.02, 'TasksToExecute', inf, 'ExecutionMode', 'FixedRate');

% start the timer
start(t);
end

% -----------------------------------------
% Callback that executes when a key is pressed
function figure_KeyPressFcn(hObject, eventdata)
    % Which key was pressed?
    the_key_pressed = eventdata.Key;
    
    % retrieve the stored motion data
    userdata = get(hObject, 'UserData');
    
    % update the velocity based on which key was pressed
    u = [0,0];
    switch the_key_pressed
        case 'downarrow'
            u(2) = -1;
        case 'uparrow'
            u(2) = 1;
        case 'leftarrow'
            u(1) = -1;
        case 'rightarrow'
            u(1) = 1;
        otherwise
            disp('unknown key');
    end

    userdata = model(userdata, u, 0.01);

    % update the stored motion data
    set(hObject, 'UserData', userdata);
end

% -----------------------------------------
% CALLBACK which moves the ball and draws itclean
function move_ball(hObject, eventdata, f)

% retrieve the stored motion data
userdata = get(f, 'UserData');

% extract the individual motion components
velocity = userdata(1:2);
position = userdata(3:4);

% is new positon within figure bounds?
% if not - then don't change the position
new_position = position + velocity;

if (abs(new_position(1)) > 100 | abs(new_position(2)) > 100)
    new_position = position;
end

%update the stored motion data
userdata(3:4) = new_position;
set(f, 'UserData', userdata);

% redraw the ball
plot(new_position(1), new_position(2),'o');
axis([-100 100 -100 100]);
title(['Velocity vector is : (', num2str(round(userdata(1) * 10) /10), ' , ', num2str(round(userdata(2) * 10)/10), ')']);
end