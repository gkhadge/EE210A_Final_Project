%simple feature extraction, grab the N largest peaks in each frame

function feature_matrix = extract_features(sound, N)
    Fs = 8000;
    framesize = 80;
    overlap = 20;
    
    %Set default number of features to 2
      if nargin < 2
        N = 2; %length of feature vector
      end
    
    NFFT = 2^nextpow2(framesize);
    
    frames = buffer(sound, framesize, overlap);
    w = hamming(framesize);
    L = size(frames,2);
    feature_matrix = zeros(N,L);
    
    i = 1;
    for frame = frames
        x = frame .* w;
        X = fft(x, NFFT)/framesize;
        freqs = (Fs/2)*linspace(0,1, NFFT/2 + 1);
        
        % Get the N largest peaks from the single-sided amplitude spectrum
        [peaks, locs] = findpeaks(abs(X(1:(NFFT/2 + 1))),'SortStr','descend');
        
        if isempty(locs)
            locs = ones(N,1);
        elseif length(locs) < N
            locs = padarray(locs, N-length(locs), 1, 'post');
        end
        
        feature_matrix(:,i) = freqs(locs(1:N))';
        i = i+1;
    end
        
    
    