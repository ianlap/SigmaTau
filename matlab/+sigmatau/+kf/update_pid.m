function [steer, pid_state] = update_pid(pid_state, x, nstates, g_p, g_i, g_d)
    % UPDATE_PID Computes the steering correction and updates PID state
    % MATLAB PID convention: integral accumulates phase error directly
    
    pid_state(1) = pid_state(1) + x(1); % sumx accumulates phase error
    
    steer = -g_p * x(1) - g_i * pid_state(1);
    
    if nstates >= 2
        steer = steer - g_d * x(2);
    end
    
    pid_state(2) = steer; % last_steer
end
