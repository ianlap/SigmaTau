function ms = default_mlist(N, min_factor)
% DEFAULT_MLIST  Octave-spaced averaging factors 1,2,4,... with N/m >= min_factor.
%
%   ms = sigmatau.util.default_mlist(N, min_factor)
%
%   Returns row vector of integer averaging factors: 2^0, 2^1, ..., 2^K
%   where K = floor(log2(N / min_factor)).
%   Matches Julia: [2^k for k in 0:floor(Int, log2(N / min_factor))]

K = floor(log2(N / min_factor));
if K < 0
    ms = zeros(1, 0, 'int32');
    return;
end
ms = 2 .^ (0:K);
end
