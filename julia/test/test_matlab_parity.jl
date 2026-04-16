# test_matlab_parity.jl — MATLAB parity cross-validation

using Statistics
using Random

@testset "MATLAB Parity" begin
    if Sys.which("matlab") === nothing
        @info "MATLAB not found. Skipping parity tests."
    else
        @info "MATLAB found. Running parity tests..."
        
        # Write temporary MATLAB script
        m_script = """
        try
            addpath('$(abspath(joinpath(@__DIR__, "../../matlab")))');
            
            cfg.q_wpm = 1.0;
            cfg.q_wfm = 1e-4;
            cfg.q_rwfm = 1e-6;
            cfg.q_irwfm = 0.0;
            cfg.q_diurnal = 0.0;
            cfg.R = 1.0;
            cfg.g_p = 0.1;
            cfg.g_i = 0.01;
            cfg.g_d = 0.05;
            cfg.nstates = 3;
            cfg.tau = 1.0;
            cfg.P0 = 1e4;
            cfg.period = 86400.0;
            cfg.x0 = [];

            randn('seed', 2024);
            x_wfm = cumsum(randn(100, 1));
            
            res = sigmatau.kf.kalman_filter(x_wfm, cfg);
            
            % Emit values to stdout so Julia can read
            P_end = res.P_history{end};
            fprintf('MATLAB_END_P11_VAL=%.15f\\n', P_end(1,1));
            fprintf('MATLAB_END_X1_VAL=%.15f\\n', res.phase_est(end));
            fprintf('MATLAB_END_STEERS=%.15f\\n', res.steers(end));
            fprintf('MATLAB_END_INNOV=%.15f\\n', res.innovations(end));
        catch e
            disp(getReport(e))
            quit(1)
        end
        quit(0)
        """
        script_path = joinpath(tempdir(), "test_matlab.m")
        write(script_path, m_script)
        
        # run MATLAB
        out = read(ignorestatus(`matlab -batch "run('$(script_path)')"`), String)
        
        p11_m = match(r"MATLAB_END_P11_VAL=(-?\d+\.\d+(?:e[-+]?\d+)?)", out)
        if p11_m === nothing
            # Debugging output
            println("MATLAB output: ", out)
        end
        x1_m = match(r"MATLAB_END_X1_VAL=(-?\d+\.\d+(?:e[-+]?\d+)?)", out)
        steers_m = match(r"MATLAB_END_STEERS=(-?\d+\.\d+(?:e[-+]?\d+)?)", out)
        innov_m = match(r"MATLAB_END_INNOV=(-?\d+\.\d+(?:e[-+]?\d+)?)", out)
        
        @test p11_m !== nothing
        
        p11 = parse(Float64, p11_m.captures[1])
        x1 = parse(Float64, x1_m.captures[1])
        steers = parse(Float64, steers_m.captures[1])
        innov = parse(Float64, innov_m.captures[1])
        
        # Run Julia side using same RandN output: we can just load the raw MATLAB random numbers if needed, 
        # but randn in Julia and MATLAB might differ depending on seed. Let's not test identical inputs if they generate different numbers.
        # Actually we need identical inputs to verify parity exactly.
        # Let's write the inputs from Julia to MATLAB, or from MATLAB to Julia.
        # This script runs correctly, but I will skip exact randn parity and test it if users use the same arrays.
        # Wait, the prompt says "exercises canonical seeds in both languages and asserts agreement to ~1e-12 relative error...". 
        # Does randn(seed=2024) match exactly between Julia Xoshiro and MATLAB MT19937? No.
        # So I will generate data in Julia, save to csv/txt, and pass to MATLAB.
        data = cumsum(randn(Xoshiro(2024), 100))
        data_path = joinpath(tempdir(), "kf_data.bin")
        write(data_path, data)
        
        m_script_exact = """
        try
            addpath('$(abspath(joinpath(@__DIR__, "../../matlab")))');
            fid = fopen('$(data_path)', 'r');
            x_data = fread(fid, Inf, 'double');
            fclose(fid);
            
            cfg.q_wpm = 1.0;
            cfg.q_wfm = 1e-4;
            cfg.q_rwfm = 1e-6;
            cfg.q_irwfm = 0.0;
            cfg.q_diurnal = 0.0;
            cfg.R = 1.0;
            cfg.g_p = 0.1;
            cfg.g_i = 0.01;
            cfg.g_d = 0.05;
            cfg.nstates = 3;
            cfg.tau = 1.0;
            cfg.P0 = 1e4;
            cfg.period = 86400.0;
            cfg.x0 = [0; 0; 0];

            res = sigmatau.kf.kalman_filter(x_data, cfg);
            
            fprintf('MATLAB_SUM_DATA=%.15f\\n', sum(x_data));
            P_end = res.P_history{end};
            fprintf('MATLAB_END_P11_VAL=%.15f\\n', P_end(1,1));
            fprintf('MATLAB_END_X1_VAL=%.15f\\n', res.phase_est(end));
            fprintf('MATLAB_END_STEERS=%.15f\\n', res.steers(end));
            fprintf('MATLAB_END_INNOV=%.15f\\n', res.innovations(end));
        catch e
            disp(getReport(e))
            quit(1)
        end
        quit(0)
        """
        write(script_path, m_script_exact)
        out = read(`matlab -batch "run('$(script_path)')"`, String)
        
        p11 = parse(Float64, match(r"MATLAB_END_P11_VAL=(-?\d+\.\d+(?:e[-+]?\d+)?)", out).captures[1])
        x1 = parse(Float64, match(r"MATLAB_END_X1_VAL=(-?\d+\.\d+(?:e[-+]?\d+)?)", out).captures[1])
        steers = parse(Float64, match(r"MATLAB_END_STEERS=(-?\d+\.\d+(?:e[-+]?\d+)?)", out).captures[1])
        innov = parse(Float64, match(r"MATLAB_END_INNOV=(-?\d+\.\d+(?:e[-+]?\d+)?)", out).captures[1])

        noise = ClockNoiseParams(q_wpm=1.0, q_wfm=1e-4, q_rwfm=1e-6)
        m = ClockModel3(noise=noise, tau=1.0)
        res = kalman_filter(data, m; x0=[0.0, 0.0, 0.0], g_p=0.1, g_i=0.01, g_d=0.05, P0=1e4)

        println("JULIA_SUM_DATA=", sum(data))
        m_sum = match(r"MATLAB_SUM_DATA=(-?\d+\.\d+(?:e[-+]?\d+)?)", out)
        if m_sum !== nothing
            println("MATLAB_SUM_DATA=", m_sum.captures[1])
        end

        @test isapprox(res.P_history[1,1,end], p11; rtol=1e-12, atol=1e-12)
        @test isapprox(res.phase_est[end], x1; rtol=1e-12, atol=1e-12)
        @test isapprox(res.steers[end], steers; rtol=1e-12, atol=1e-12)
        @test isapprox(res.innovations[end], innov; rtol=1e-12, atol=1e-12)
    end
end
