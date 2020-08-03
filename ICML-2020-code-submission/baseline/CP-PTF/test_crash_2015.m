clear all;
close all;
rng('default');

addpath(genpath('../util'));
addpath(genpath('../tensor_toolbox_2.6'));
addpath(genpath('../lightspeed'));
addpath(genpath('../minFunc_2012'));
% addpath('../');

load('../mat/crash_2015.mat');

data.ind = double(train_ind) + 1;
data.e = train_y;

nvec = max(data.ind);
nmod = size(nvec,2);
data.e_count = sptensor(data.ind(1,:), 1, nvec);
for n=2:size(data.ind,1)
    sub = data.ind(n,:);
    data.e_count(sub) = data.e_count(sub) + 1;
end
data.T = max(data.e);

data.train_subs = find(data.e_count);
data.y_subs = data.e_count(data.train_subs);
data.tensor_sz = nvec;

test.ind = double(test_ind) + 1;
test.e = test_y;

% test = test.data;
test.e_count = sptensor(test.ind(1,:), 1, nvec);
for n=2:size(test.ind,1)
    sub = test.ind(n,:);
    test.e_count(sub) = test.e_count(sub) + 1;
end
test.T = max(test.e)-min(test.e);
data.test_ind = find(test.e_count);
data.test_vals = test.e_count(data.test_ind);
data.test_T = test.T;

model = [];
model.R = 1;
for k=1:nmod
    model.U{k} = rand(nvec(k), model.R);
end
%decay rate (w.r.t. AdDelta)
model.decay = 0.95;
%no. of epoch
model.epoch = 100;
%batch of events
model.batch_size = 100;
%training time period
model.T = data.T; 
%training
[model,test_ll] = CPTensorPP_online_robust(data, model);

%test
t_ll = predictive_log_likelihood(model, data);
fprintf('CP-PP Tensor, test log-likelihood: %g\n', t_ll);
% save('../plot/CP-PP-ufo.mat', 'test_ll');

