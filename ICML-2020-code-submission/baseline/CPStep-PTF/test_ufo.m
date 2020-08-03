clear all;
close all;
rng('default');

addpath(genpath(('../util')));
addpath(genpath(('../tensor_toolbox_2.6')));
addpath(genpath(('../lightspeed')));
addpath(genpath(('../minFunc_2012')));
% addpath('../');

%core input files
% train_file = '../data/ufo-train-hybrid-1-300.mat';
% test_file = '../data/ufo-test-hybrid-1-300.mat';

% R = 2;
time_steps = 3;%5 10, 20, 30
% model_file = strcat('./cp-markov-pp-ufo-R-', num2str(R), '-TimeStep-', num2str(time_steps), '-models.mat');
% res_file =   strcat('./cp-markov-pp-ufo-R-', num2str(R), '-TimeStep-', num2str(time_steps), '-all.mat');
decay = 0.95;
nepoch = 100;



load('../mat/ufo.mat');
data.ind = double(train_ind) + 1;
data.e = train_y;
nvec = max(data.ind);
data.T = max(data.e);
%time interval
Dmax = data.T/time_steps;
%Dmax = data.T/30;
num_t = floor(data.T/Dmax);
nvec = [nvec, num_t];
nmod = size(nvec,2);

data.e_count = sptensor([data.ind(1,:),1], 1, nvec);
for n=2:size(data.ind,1)
    t_index = max(1, min(ceil(data.e(n)/Dmax), num_t));    
    sub = [data.ind(n,:), t_index];
    data.e_count(sub) = data.e_count(sub) + 1;
end

data.train_subs = find(data.e_count);
data.y_subs = data.e_count(data.train_subs);
data.tensor_sz = nvec;

% test = load(test_file);
% test = test.data;
test.ind = double(test_ind) + 1;
test.e = test_y;
%prediction using the last time factor
test.e_count = sptensor([test.ind(1,:), num_t], 1, nvec);
for n=2:size(test.ind,1)
    sub = [test.ind(n,:), num_t];
    test.e_count(sub) = test.e_count(sub) + 1;
end
test.T = max(test.e)-min(test.e);
data.test_ind = find(test.e_count);
data.test_vals = test.e_count(data.test_ind);
data.test_T = test.T;


%R_list = [1, 2, 5, 8 10];
% time_list = [5, 10, 20];
R_list =[2,5,8,10];
res = [];

% for ti = 1: 3
    for ri = 1: 4
        R = R_list(ri);
        model = [];
        model.R = R;
        for k=1:nmod
            %model.U{k} = randn(nvec(k), model.R);
            model.U{k} = rand(nvec(k), model.R);
            %model.U{k} = 0.1*randn(nvec(k), model.R);
        end
        %decay rate (w.r.t. AdDelta)
        model.decay = decay;
        %no. of epoch
        model.epoch = nepoch;
        %batch of events
        model.batch_size = 100;
        %training time period
        model.T = Dmax; 
        %training
        [models,test_ll] = CPMarkovPP_online_robust_advanced(data, model);
% 
%         save(model_file, 'models');
%         save(res_file, 'test_ll');
%         plot(1:nepoch, test_ll);
        res(ri) = max(test_ll);
    end
res
% end


% model_file
% res_file


