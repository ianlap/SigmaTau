function x = generate(alpha, N, tau0)
% GENERATE  Generate power-law noise using the Kasdin FFT method.
%
%   x = sigmatau.noise.generate(alpha, N)
%   x = sigmatau.noise.generate(alpha, N, tau0)
%
%   Inputs:
%     alpha – frequency noise exponent (Sy(f) ~ f^alpha):
%               2 = White PM, 0 = White FM, -2 = RWFM
%     N     – number of output phase samples (must be even)
%     tau0  – sampling interval in seconds (default: 1.0)
%
%   Output:
%     x – phase data column vector of length N
%
%   Algorithm:
%     1. Shape white Gaussian noise in the frequency domain: amplitude ~ f^(alpha/2)
%     2. IFFT → frequency data; normalise to unit std
%     3. cumsum * tau0 → phase data
%
%   Note: Set rng before calling for reproducible output (e.g. rng(42)).
%
%   Reference:
%     N. J. Kasdin & T. Walter, "Discrete simulation of power law noise",
%     Proc. IEEE Frequency Control Symposium, pp. 274-283, 1992.

if nargin < 3, tau0 = 1.0; end

if mod(N, 2) ~= 0
    error('SigmaTau:generate', 'N must be even for FFT symmetry.');
end

% Frequency bins 1 .. N/2 (normalised; DC excluded)
f = (1:N/2)';

% Magnitude spectrum: amplitude ~ f^(alpha/2)
S_f = f .^ (alpha / 2);

% Random phases for bins 1 .. N/2-1 (DC=0 and Nyquist=0 forced real)
phase = 2 * pi * rand(N/2 - 1, 1);

% Build one-sided spectrum
half              = zeros(N/2 + 1, 1);
half(1)           = 0;                              % DC = 0
half(2:N/2)       = S_f(1:end-1) .* exp(1j*phase); % positive freqs
half(N/2 + 1)     = 0;                              % Nyquist = 0

% Mirror to conjugate-symmetric full spectrum
full = [half; conj(flipud(half(2:N/2)))];

% IFFT → time-domain frequency data
x_freq = real(ifft(full));

% Normalise frequency data to unit std (SP1065 convention)
x_freq = x_freq - mean(x_freq);
x_freq = x_freq / std(x_freq);

% Integrate frequency → phase; scale by tau0
x = cumsum(x_freq) * tau0;
end
