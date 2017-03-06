classdef Word < handle %inherit from handle so all copies reference this one class
    properties
       %add more properties as needed
       N     =   2; % number of states (adjustable variable)
       A     =  []; % NxN transition probability matrix (fill out with baum welch)
       prior =  []; % Nx1 initial state distribution vector
       mu    =  []; % DxN mean vector (D = number of features) 
       Sigma =  []; % DxDxN covariance matrix
       name  =  ''; % Word label
    end
    
    methods
        function self = Word(name)
            self.name = char(name);
        end
        
        % for testing purposes, will set prior with baum welch training 
        function setPrior(self,prior)
            self.prior = prior; 
        end
        
        % for testing purposes, will set TPM with baum welch training
        function setA(self,A)
            self.A = A;
        end
        
        function log_likelihood = log_likelihood(self, observations)
            D = self.state_likelihood(observations);
            [~, c] = self.forward(observations, D);
            log_likelihood = sum(log(c));
        end
        
        % observations refers to just 1 set of observations
        % each word will have multiple sets of observations, 1 per audio
        % file
        
        % need f(o|state) at each observation step for each state,
        % this will be represented in matrix D (size NxL, L length of
        % observations)
        function [alpha, c] = forward(self, observations, D, normalize)
            if nargin < 4
                normalize = 1;
            end
			
            L = size(observations,2); % num of observations
            alpha = zeros(self.N,L);
            c = ones(L,1);
			
            for i = 1:L
                if i == 1                               %initialize 
                    alpha(:,i) = D(:,i).*self.prior;
                else                                    %recursion
                    alpha(:,i) = D(:,i).*(self.A'*alpha(:,i-1));
                end
                
                if (normalize)
					c(i) = sum(alpha(:,i));
                    alpha(:,i) = alpha(:,i)/c(i); 
                end
            end
        end
        
        %calculate backwards recursion (no normalization)
        function beta = backward(self, observations, D)
            L = size(observations,2); 
            beta = zeros(self.N,L);
            
            for i = L:-1:1
                if i == L                               %initialization
                    beta(:,i) = ones(size(beta(:,i)));
                else                                    %recursion
                    beta(:,i) = self.A*(D(:,i+1).*beta(:,i+1));
                end
            end 
        end
        
        %do both forward and backwards recursion with normalization
        function [alpha, beta, c] = forward_backward(self, observations, D)
            [alpha, c] = self.forward(observations, D);
            
            %now do the backwards part with normalization
            L = size(observations,2); 
            beta = zeros(self.N,L);
            
            for i = L:-1:1
                if i == L                               %initialization
                    beta(:,i) = ones(size(beta(:,i)));
                else                                    %recursion
                    beta(:,i) = (self.A*(D(:,i+1).*beta(:,i+1)))/c(i+1);
                end
            end 
        end
		
		% create the D matrix based on observations and current state of 
		% mu and Sigma (assuming gaussian mixture model)
		function D = state_likelihood(self, observations)
            D = zeros(self.N, size(observations,2));
            
            for i = 1:self.N
                D(i, :) = mvnpdf(observations', self.mu(:,i)', self.Sigma(:,:,i));
            end
		end

        % perform baum welch training here. Will set prior, and estimate A,
        % mu, and Sigma
		% implement assuming 1 set of observations first, extend to multiple sets later
        % r NxL (num states x length of observation)
        % S NxNxL-1 (L-1 NxN matrices)
        % Nr Nx1
        function [r,S,Nr] = e_step(self, observations)
            D = self.state_likelihood(observations);
            [alpha, beta, c] = self.forward_backward(observations, D);
            L = size(observations,2);
            
            r = alpha.*beta;            %size NxL (L length of observation vector, N number of states)
            S = zeros(self.N, self.N, L-1);
            for i = 2:L
                S(:,:,i-1) = ((self.A).*(alpha(:,i-1)*(beta(:,i).*D(:,i))'))/c(i);
            end
            Nr = sum(r,2);
            
        end
		
		% observation set will probably have to be a cell matrix since each set of 
		% observations can be a different length
		function trainAll(self, observation_set, num_iter)
            L = size(observation_set,2);

            for i = 1:num_iter
                expected_mu = zeros(size(self.mu));     %size DxN (D size of feature vector)
                expected_Sigma = zeros(size(self.Sigma));
                expected_prior_num = zeros(size(self.prior));
                expected_prior_den = 0;
                expected_A_num = zeros(size(self.A));
                %expected_A_den = zeros(size(self.A));
                % since we know A is a stochastic matrix, we know the rows
                % must sum up to 1. Since we know the numerator, we can
                % just normalize the rows instead
                expected_N = zeros(self.N,1);
                
                for l = 1:L
                    Y = observation_set{l};
                    Nl = size(Y,2);
                    num_features = size(Y,1);
                  	[r,S,Nr] = self.e_step(Y);
                    for j = 1:self.N
                        mu_k = repmat(self.mu(:,j),1,Nl); %[mu_k mu_k ... mu_k]
                        r_k = repmat(r(j,:),num_features,1); 
                        expected_mu(:,j) = expected_mu(:,j) + sum(r_k.*Y,2);
                        expected_Sigma(:,:,j) = expected_Sigma(:,:,j) + (r_k.*(Y-mu_k))*(Y-mu_k)';
                    end
                    expected_prior_num = expected_prior_num + r(:,1);
                    expected_prior_den = expected_prior_den + sum(r(:,1));
                    expected_A_num = expected_A_num + sum(S,3);
                    %expected_A_den = expected_A_den + repmat(sum(sum(S,3)),self.N,1);
                    expected_N = expected_N + Nr;    
                end
                
                % Set any zeros to one before dividing to avoid NaN
                expected_N = expected_N + (expected_N == 0);
                
                for j = 1:self.N
                    self.mu(:,j) = expected_mu(:,j)/expected_N(j);
                    self.Sigma(:,:,j) = expected_Sigma(:,:,j)/expected_N(j);
                end
                self.prior = expected_prior_num/expected_prior_den;
                self.A = normalize_rows(expected_A_num);
            end
        end
        
        function initialize(self, observations)
            % Random prior and random A matrix
            self.prior = rand(self.N, 1);
            self.prior = self.prior/(sum(self.prior));
            % Can also try uniform priors
            % self.prior = ones(self.N, 1)/self.N;
            self.A = normalize_rows(rand(self.N));

            % Use one set of observations to form a diagonal covariance
            self.Sigma = repmat(diag(diag(cov(observations'))), [1 1 self.N]);
            
            % Pick a random data points to be the mean
            indices = randperm(size(observations, 2));
            self.mu = observations(:, indices(1:self.N));
        end
        %add functions as needed
      
        
    end
end