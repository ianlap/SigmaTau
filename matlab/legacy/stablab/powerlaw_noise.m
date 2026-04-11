%POWERLAW_NOISE  Generate power-law noise with given alpha using FFT
%
%   x = powerlaw_noise(alpha, N)
%
%   Generates a 1D time series of length N with power spectral density
%   S(f) ~ f^alpha using frequency-domain shaping (Kasdin method).
%
%   Inputs:
%     alpha - power-law exponent (S_y(f) ~ f^alpha), typically in [-2, 2]
%     N     - number of points in time series (should be power of 2 for speed)
%
%   Output:
%     x     - generated time series (real-valued)
%
%   Reference:
%     N. J. Kasdin and T. Walter, "Discrete simulation of power law noise",
%     Proc. 1992 IEEE Frequency Control Symposium, pp. 274–283, May 1992.

function x = powerlaw_noise(alpha, N)
    if nargin < 2
        error('Usage: x = powerlaw_noise(alpha, N)');
    end
    if mod(N,2) ~= 0
        error('N must be even for FFT symmetry.');
    end

    % Frequency vector (normalized to sampling rate)
    f = (1:N/2)';  % exclude DC (f=0)

    % Desired power spectral density slope
    S_f = f .^ (alpha/2);  % magnitude spectrum ~ f^(alpha/2)

    % Random phases for positive freqs (except DC and Nyquist)
    rng('shuffle');
    phase = 2*pi*rand(N/2-1, 1);

    % Complex spectrum (DC + freqs 1:N/2-1 + Nyquist)
    half_spectrum = zeros(N/2+1, 1);
    half_spectrum(2:N/2) = S_f(1:end-1) .* exp(1j*phase); % positive freqs
    half_spectrum(1) = 0;              % DC component
    half_spectrum(N/2+1) = 0;          % Nyquist freq = real only

    % Mirror for negative freqs (conjugate symmetric)
    full_spectrum = [half_spectrum; conj(flipud(half_spectrum(2:N/2)))];

    % IFFT and take real part
    x = real(ifft(full_spectrum));

    % Normalize to unit standard deviation
    x = x - mean(x);
    x = x / std(x);
    x = cumsum(x); %convert to phase data
end
