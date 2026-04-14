function x = predict_state(x, Phi, pid_state_last_steer, tau, nstates)
    % PREDICT_STATE Advance the state vector and apply steering corrections
    
    x = Phi * x;
    
    % phase correction from last steer
    x(1) = x(1) + pid_state_last_steer * tau;
    
    if nstates >= 2
        % freq correction
        x(2) = x(2) + pid_state_last_steer;
    end
end
