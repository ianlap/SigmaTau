function Phi = build_phi(nstates, tau)
    % BUILD_PHI State transition matrix
    % Phi encodes constant-velocity/acceleration kinematics over interval tau
    
    Phi = zeros(nstates, nstates);
    for i = 1:nstates
        Phi(i, i) = 1.0;
    end
    
    if nstates >= 2
        Phi(1, 2) = tau;
    end
    
    if nstates >= 3
        Phi(1, 3) = tau^2 / 2.0;
        Phi(2, 3) = tau;
    end
    
    % nstates == 5: diurnal states 4,5 are identity (no kinematic coupling)
end
