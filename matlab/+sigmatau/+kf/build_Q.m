function Q = build_Q(nstates, q_wfm, q_rwfm, q_irwfm, q_diurnal, tau)
    % BUILD_Q Process noise matrix
    % SP1065: continuous-time noise model integrated over [0, tau]
    
    Q = zeros(nstates, nstates);
    tau2 = tau^2;
    tau3 = tau^3;
    tau4 = tau^4;
    tau5 = tau^5;
    
    Q(1, 1) = q_wfm * tau + q_rwfm * tau3 / 3.0 + q_irwfm * tau5 / 20.0;
    
    if nstates >= 2
        Q(1, 2) = q_rwfm * tau2 / 2.0 + q_irwfm * tau4 / 8.0;
        Q(2, 1) = Q(1, 2);
        Q(2, 2) = q_rwfm * tau + q_irwfm * tau3 / 3.0;
    end
    
    if nstates >= 3
        Q(1, 3) = q_irwfm * tau3 / 6.0;
        Q(3, 1) = Q(1, 3);
        Q(2, 3) = q_irwfm * tau2 / 2.0;
        Q(3, 2) = Q(2, 3);
        Q(3, 3) = q_irwfm * tau;
    end
    
    if nstates == 5
        Q(4, 4) = q_diurnal;
        Q(5, 5) = q_diurnal;
    end
end
