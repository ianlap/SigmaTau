function P = predict_covariance(P, Phi, Q)
    % PREDICT_COVARIANCE Advance the error covariance matrix
    
    Pm = Phi * P * Phi' + Q;
    % Ensure symmetry
    P = (Pm + Pm') / 2.0;
end
