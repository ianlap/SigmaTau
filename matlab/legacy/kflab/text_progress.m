function text_progress(progress, label)
% TEXT_PROGRESS - Simple text-based progress indicator
%
% Usage:
%   text_progress(0, 'Processing')      % Initialize with label
%   text_progress(0.5)                  % Update to 50%
%   text_progress(1)                    % Complete (adds newline)
%
% Displays: "Processing: ....." for 50% progress

persistent current_label last_dots

if nargin == 2
    % Initialize with new label
    current_label = label;
    last_dots = 0;
    fprintf('%s: ', label);
    return;
end

if nargin == 1 && ~isempty(current_label)
    % Update progress
    n_dots = floor(progress * 10);  % 10 dots for 100%
    
    % Print new dots since last update
    new_dots = n_dots - last_dots;
    if new_dots > 0
        fprintf(repmat('.', 1, new_dots));
        last_dots = n_dots;
    end
    
    % Add newline when complete
    if progress >= 1
        fprintf(' Done\n');
        current_label = '';
        last_dots = 0;
    end
end

end