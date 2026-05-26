%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a script to run the baseline methods on %
% different datasets and parameters.              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc
warning off;

path = './';
addpath(genpath(path));

%% Load data
dataPath = './datasets/';

% Settings
dataName = {'Caltech101_7'};
IMvSSL_METHODS = {'SLIM', 'AMSC'};
k = 7;
test_times = 10;
missing_rate_list = [0.3, 0.5, 0.7];
labeled_rate_list = [0.05];
ANCHOR_EXP_NUM = struct('Caltech101_7', 7);
METHODS = {'SLIM', 'AMSC', 'FMSSL', 'AMUSE', 'ERL_MVSC', ...
            'MVAR', 'FMSEL', 'CFSMC', 'AMMSS', 'AMGL', 'MLAN'};


[~,MvSSL_idx] = ismember(IMvSSL_METHODS, METHODS);
numExist = sum(MvSSL_idx > 0);

if length(METHODS) - numExist > 0
    USE_DMF_COMPLETION = 1;
else
    USE_DMF_COMPLETION = 0;
end


%% MAIN
for ds_i = 1:length(dataName)
    ds_name = dataName{ds_i};
    fprintf('Dataset: %s\n', ds_name);
    for mr_i = 1:length(missing_rate_list)
        for lr_i = 1:length(labeled_rate_list)
            disp("This script is running baselines:");
            disp(METHODS);

            missing_rate = missing_rate_list(mr_i);
            labeled_rate = labeled_rate_list(lr_i);

            load([dataPath, ds_name, '.mat'], "X", "gnd");
            Y = gnd;
            num_views = length(X);
            num_samples = size(X{1}, 1);
            num_class = length(unique(Y));
            anchor_exp = ANCHOR_EXP_NUM.(ds_name);
            

            Time_res = struct(); ACC_res = struct();
            Fscore_res = struct(); Prec_res = struct();
            best_acc = 0; best_f1 = 0; best_prec = 0;
            best_rho = 0; best_beta = 0; best_time = 0;
            for test_i = 1:test_times
                fprintf('Test %d\n', test_i);
                %% Generate partial data
                [M, temp_index] = partialData(X, missing_rate, 2);
                Y = reshape(Y, [], 1); % make sure Y is a column vector
                [MX, Ll, Lu, existing_index, ic] = labelData(M, Y, temp_index, labeled_rate);
                num_labeled = size(Ll, 1);
                % params.Lu = Lu; % for training time evaluation

                %% DMF completion
                if USE_DMF_COMPLETION
                    [Xi, O, dim] = cdata(M, temp_index);
                    s = [num_class 10*num_class sum(dim)];
                    options.maxiter = 500;
                    options.activation_func = {'tanh_opt','linear'};
                    options.Wp = 0.01;
                    options.Zp = 0.01;
                    [Xr,~] = MC_DMF(Xi,O,s,options);
                    [XD] = sdata(Xr,dim);
                    [MD] = bbcselectdata(XD, ic, num_views); % recovered dataset
                    clear XD Xr;
                end

                %% Construct the partial graph (return tensor_Z)
                mode = 1; % 1 for "BKHK"

                for v = 1:num_views
                    V_existing_ind = find(existing_index(:, v) == 1);
                    if 2^anchor_exp > length(V_existing_ind)
                        num_anchor = 2^(floor(log2(length(V_existing_ind))));
                    else
                        num_anchor = 2^anchor_exp;
                    end
                end

                if USE_DMF_COMPLETION
                    [bA] = csgraphanchor(MD, num_anchor, k);
                end

                %% Baselines
                for method_i = 1:length(METHODS)
                    switch METHODS{method_i}
                        case 'SLIM'
                            lam1 = [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4];
                            lam2 = [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4];
                            if test_i == 1
                                T = 0;
                                for lam1_i = 1:length(lam1)
                                    for lam2_i = 1:length(lam2)
                                        options.lam1 = lam1(lam1_i);
                                        options.lam2 = lam2(lam2_i);
                                        options.stop = 1e-4;
                                        tic
                                        [Yslim,~,~] = SLIM(MX,Ll,existing_index,options);
                                        temp_toc = toc;
                                        [ACC, F1] = accfscore(Yslim, Lu);
                                        [~,~,Prec] = calcMultiClassScore(Yslim, Lu, num_class);
                                        if ACC+F1 > T
                                            SLIM_best_param.lam1 = options.lam1;
                                            SLIM_best_param.lam2 = options.lam2;
                                            Time_res.(METHODS{method_i})(test_i) = temp_toc;
                                            ACC_res.(METHODS{method_i})(test_i) = ACC;
                                            Fscore_res.(METHODS{method_i})(test_i) = F1;
                                            Prec_res.(METHODS{method_i})(test_i) = Prec;
                                            T = ACC + F1;
                                        end
                                    end
                                end
                            else
                                options.lam1 = SLIM_best_param.lam1;
                                options.lam2 = SLIM_best_param.lam2;
                                options.stop = 1e-4;
                                tic
                                [Yslim,~,~] = SLIM(MX,Ll,existing_index,options);
                                temp_time = toc;
                                [ACC, F1] = accfscore(Yslim, Lu);
                                [~,~,Prec] = calcMultiClassScore(Yslim, Lu, num_class);
                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                            end
                            clear Yslim;

                        case 'AMSC'
                            for v = 1:num_views
                                m(v) = num_anchor;
                            end
                            Z = partialanchors(MX, existing_index, m, k);
                            if test_i == 1
                                T = 0;
                                pmssParam.p = 0.5;
                                for r = 1.1:0.5:3.1
                                    pmssParam.r = r;
                                    tic
                                    [Ypmss,~] = pmss_Woodbury(Ll, existing_index, Z, pmssParam);
                                    temp_time = toc;
                                    [ACC, F1] = accfscore(Ypmss, Lu);
                                    [~,~,Prec] = calcMultiClassScore(Ypmss, Lu, num_class);
                                    if ACC+F1 > T
                                        AMSC_best_param.r = pmssParam.r;
                                        Time_res.(METHODS{method_i})(test_i) = temp_time;
                                        ACC_res.(METHODS{method_i})(test_i) = ACC;
                                        Fscore_res.(METHODS{method_i})(test_i) = F1;
                                        Prec_res.(METHODS{method_i})(test_i) = Prec;
                                        T = ACC + F1;
                                    end
                                end
                            else
                                pmssParam.p = 0.5;
                                pmssParam.r = AMSC_best_param.r;
                                tic
                                [Ypmss,~] = pmss_Woodbury(Ll, existing_index, Z, pmssParam);
                                temp_time = toc;
                                [ACC, F1] = accfscore(Ypmss, Lu);
                                [~,~,Prec] = calcMultiClassScore(Ypmss, Lu, num_class);
                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                            end
                            clear Ypmss Z;

                        case 'FMSSL'
                            if num_samples < 10000
                                fmmsl_kcNumAnchors = 8; % 2^{NumAnchors}
                            else
                                fmmsl_kcNumAnchors = 10; % 2^{NumAnchors}
                            end
                            fmmsl_kcNumNeighbor = k;
                            fmssl_rho=1.2;
                            fmssl_u=999;
                            fmssl_mu=1; graph_input=0;
                            if test_i == 1
                                T = 0;
                                for fmssl_a = -4:2:2
                                    alpha_fmssl = 10^fmssl_a;
                                    tic
                                    [Yfmssldmf,~] = myFMSSL(MD,Ll,num_class,fmmsl_kcNumAnchors,...
                                                        fmmsl_kcNumNeighbor,alpha_fmssl,...
                                                        fmssl_u,fmssl_mu,fmssl_rho,graph_input);
                                    temp_time = toc;
                                    [ACC, F1] = accfscore(Yfmssldmf, Lu);
                                    [~,~,Prec] = calcMultiClassScore(Yfmssldmf, Lu, num_class);
                                    if ACC+F1 > T
                                        FMSSL_best_param.alpha_fmssl = alpha_fmssl;
                                        Time_res.(METHODS{method_i})(test_i) = temp_time;
                                        ACC_res.(METHODS{method_i})(test_i) = ACC;
                                        Fscore_res.(METHODS{method_i})(test_i) = F1;
                                        Prec_res.(METHODS{method_i})(test_i) = Prec;
                                        T = ACC + F1;
                                    end
                                end
                            else
                                alpha_fmssl = FMSSL_best_param.alpha_fmssl;
                                tic
                                [Yfmssldmf,~] = myFMSSL(MD,Ll,num_class,fmmsl_kcNumAnchors,...
                                                        fmmsl_kcNumNeighbor,alpha_fmssl,...
                                                        fmssl_u,fmssl_mu,fmssl_rho,graph_input);
                                temp_time = toc;
                                [ACC, F1] = accfscore(Yfmssldmf, Lu);
                                [~,~,Prec] = calcMultiClassScore(Yfmssldmf, Lu, num_class);
                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                            end
                            clear Yfmssldmf;


                        case 'AMUSE'
                            cell_A = cell(num_views, 1); Xl = cell(num_views, 1); Xu = cell(num_views, 1);
                            for v = 1:num_views
                                Xl{v} = MD{v}(1:num_labeled, :);
                                Xu{v} = MD{v}(num_labeled+1:end, :);
                                cell_A{v} = constructW_PKN(MD{v}', k, 0);
                            end
                            Yl = TransLabelR(Ll);
                            if test_i == 1
                                T = 0;
                                for amuse_lam_i = 1:2
                                    AMUSE_lambda = 10^amuse_lam_i;
                                    tic
                                    [Yamuse, ~] = AMUSE(Xl, Xu, Yl, AMUSE_lambda, cell_A);
                                    temp_time = toc;
                                    [ACC, F1] = accfscore(Yamuse, Lu);
                                    [~,~,Prec] = calcMultiClassScore(Yamuse, Lu, num_class);
                                    if ACC+F1 > T
                                        AMUSE_best_param.AMUSE_lambda = AMUSE_lambda;
                                        Time_res.(METHODS{method_i})(test_i) = temp_time;
                                        ACC_res.(METHODS{method_i})(test_i) = ACC;
                                        Fscore_res.(METHODS{method_i})(test_i) = F1;
                                        Prec_res.(METHODS{method_i})(test_i) = Prec;
                                        T = ACC + F1;
                                    end
                                end
                            else
                                AMUSE_lambda = AMUSE_best_param.AMUSE_lambda;
                                tic
                                [Yamuse, ~] = AMUSE(Xl, Xu, Yl, AMUSE_lambda, cell_A);
                                temp_time = toc;
                                [ACC, F1] = accfscore(Yamuse, Lu);
                                [~,~,Prec] = calcMultiClassScore(Yamuse, Lu, num_class);
                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                            end
                            clear Yamuse Xl Xu cell_A Yl;

                        
                        case 'ERL_MVSC'
                            tic
                            [Yerl_mvsc] = mySemiERLMVSC(MD, Ll, num_samples, num_class);
                            temp_time = toc;
                            [ACC, F1] = accfscore(Yerl_mvsc, Lu);
                            [~,~,Prec] = calcMultiClassScore(Yerl_mvsc, Lu, num_class);
                            Time_res.(METHODS{method_i})(test_i) = temp_time;
                            ACC_res.(METHODS{method_i})(test_i) = ACC;
                            Fscore_res.(METHODS{method_i})(test_i) = F1;
                            Prec_res.(METHODS{method_i})(test_i) = Prec;
                            clear Yerl_mvsc;


                        case 'MVAR'
                            mvardmf_r = 2;
                            mvardmf_lambda = 1e2*ones(length(MD),1);
                            if test_i == 1
                                T = 0;
                                for mu_i = 0:2:6
                                    tic
                                    [Ymvardmf, ~] = myMVAR(MD,Ll,num_class,mvardmf_lambda,mu_i,mvardmf_r);
                                    temp_time = toc;
                                    [ACC, F1] = accfscore(Ymvardmf, Lu);
                                    [~,~,Prec] = calcMultiClassScore(Ymvardmf, Lu, num_class);
                                    if ACC+F1 > T
                                        MVAR_best_param.mu_i = mu_i;
                                        Time_res.(METHODS{method_i})(test_i) = temp_time;
                                        ACC_res.(METHODS{method_i})(test_i) = ACC;
                                        Fscore_res.(METHODS{method_i})(test_i) = F1;
                                        Prec_res.(METHODS{method_i})(test_i) = Prec;
                                        T = ACC + F1;
                                    end
                                end
                            else
                                tic
                                [Ymvardmf, ~] = myMVAR(MD,Ll,num_class,mvardmf_lambda,...
                                                            MVAR_best_param.mu_i,mvardmf_r);
                                temp_time = toc;
                                [ACC, F1] = accfscore(Ymvardmf, Lu);
                                [~,~,Prec] = calcMultiClassScore(Ymvardmf, Lu, num_class);
                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                            end
                            clear Ymvardmf;


                        case 'FMSEL'
                            Xl = cell(num_views,1); Xu = cell(num_views,1);
                            for v = 1:num_views
                                Xl{v} = MD{v}(1:num_labeled, :);
                                Xu{v} = MD{v}(num_labeled+1:end, :);
                                cell_A{v} = constructW_PKN(MD{v}', k, 0);
                            end
                            Yl = TransLabelR(Ll);

                            if test_i == 1
                                fmsel_lambda1_set = [1e-1];
                                fmsel_lambda2_set = [1e-1; 1e1];
                                fmsel_sigma_set = [1e-3; 1e-1];
                                T = 0;
                                for p1 = 1:length(fmsel_lambda1_set)
                                    for p2 = 1:length(fmsel_lambda2_set)
                                        for p3 = 1:length(fmsel_sigma_set)
                                            fmsel_lambda1 = fmsel_lambda1_set(p1);
                                            fmsel_lambda2 = fmsel_lambda2_set(p2);
                                            fmsel_sigma = fmsel_sigma_set(p3);
                                            tic
                                            [~,~,Yfmsel,~] = myFMSEL(Xl, Xu, Yl, fmsel_lambda1,...
                                                fmsel_lambda2, fmsel_sigma);
                                            temp_time = toc;
                                            [ACC, F1] = accfscore(Yfmsel, Lu);
                                            [~,~,Prec] = calcMultiClassScore(Yfmsel, Lu, num_class);
                                            if ACC+F1 > T
                                                FMSEL_best_param.fmsel_lambda1 = fmsel_lambda1;
                                                FMSEL_best_param.fmsel_lambda2 = fmsel_lambda2;
                                                FMSEL_best_param.fmsel_sigma = fmsel_sigma;
                                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                                                T = ACC + F1;
                                            end
                                        end
                                    end
                                end
                            else
                                tic
                                [~,~,Yfmsel,~] = myFMSEL(Xl, Xu, Yl, FMSEL_best_param.fmsel_lambda1,...
                                    FMSEL_best_param.fmsel_lambda2, FMSEL_best_param.fmsel_sigma);
                                temp_time = toc;
                                [ACC, F1] = accfscore(Yfmsel, Lu);
                                [~,~,Prec] = calcMultiClassScore(Yfmsel, Lu, num_class);
                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                            end
                            clear Yfmsel Xl Xu Yl;

                        
                        case 'CFSMC'
                            Cl=10000;
                            Cu=0.0001;
                            Xl = cell(num_views,1); Xu = cell(num_views,1);
                            for v = 1:num_views
                                Xl{v} = MD{v}(1:num_labeled, :);
                                Xu{v} = MD{v}(num_labeled+1:end, :);
                            end
                            Yl = TransLabelR(Ll);

                            if test_i == 1
                                cfsmc_lambda_set = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
                                cfsmc_gamma_set = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
                                cfsmc_beta_set = [0.001, 0.01, 0.1, 1, 10, 100, 1000];
                                T = 0;
                                for p1 = 1:length(cfsmc_lambda_set)
                                    for p2 = 1:length(cfsmc_gamma_set)
                                        for p3 = 1:length(cfsmc_beta_set)
                                            cfsmc_lambda = cfsmc_lambda_set(p1);
                                            cfsmc_gamma = cfsmc_gamma_set(p2);
                                            cfsmc_beta = cfsmc_beta_set(p3);
                                            tic
                                            [~,~,~,Ycfsmc] = CFSMC(Xl, Xu, Yl, cfsmc_lambda,...
                                                cfsmc_gamma, cfsmc_beta, Cl, Cu);
                                            temp_time = toc;
                                            [ACC, F1] = accfscore(Ycfsmc, Lu);
                                            [~,~,Prec] = calcMultiClassScore(Ycfsmc, Lu, num_class);
                                            if ACC+F1 > T
                                                CFSMC_best_param.cfsmc_lambda = cfsmc_lambda;
                                                CFSMC_best_param.cfsmc_gamma = cfsmc_gamma;
                                                CFSMC_best_param.cfsmc_beta = cfsmc_beta;
                                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                                                T = ACC + F1;
                                            end
                                        end
                                    end
                                end
                            else
                                cfsmc_lambda = CFSMC_best_param.cfsmc_lambda;
                                cfsmc_gamma = CFSMC_best_param.cfsmc_gamma;
                                cfsmc_beta = CFSMC_best_param.cfsmc_beta;
                                tic
                                [~,~,~,Ycfsmc] = CFSMC(Xl, Xu, Yl, cfsmc_lambda,...
                                    cfsmc_gamma, cfsmc_beta, Cl, Cu);
                                temp_time = toc;
                                [ACC, F1] = accfscore(Ycfsmc, Lu);
                                [~,~,Prec] = calcMultiClassScore(Ycfsmc, Lu, num_class);
                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                            end
                            clear Ycfsmc Xl Xu Yl;


                        case 'AMMSS'
                            if test_i == 1
                                T = 0;
                                for t = 0.1:0.4:2
                                    for lam = 0:0.3:1
                                        inPara.r = 10^t;
                                        inPara.lambda = lam;
                                        tic
                                        [Yammss, ~] = weightMMSSL(bA, Ll, inPara);
                                        temp_time = toc;
                                        [ACC, F1] = accfscore(Yammss, Lu);
                                        [~,~,Prec] = calcMultiClassScore(Yammss, Lu, num_class);
                                        if ACC+F1 > T
                                            AMMSS_best_param.r = inPara.r;
                                            AMMSS_best_param.lambda = inPara.lambda;
                                            Time_res.(METHODS{method_i})(test_i) = temp_time;
                                            ACC_res.(METHODS{method_i})(test_i) = ACC;
                                            Fscore_res.(METHODS{method_i})(test_i) = F1;
                                            Prec_res.(METHODS{method_i})(test_i) = Prec;
                                            T = ACC + F1;
                                        end
                                    end
                                end
                            else
                                inPara.r = AMMSS_best_param.r;
                                inPara.lambda = AMMSS_best_param.lambda;
                                tic
                                [Yammss, ~] = weightMMSSL(bA, Ll, inPara);
                                temp_time = toc;
                                [ACC, F1] = accfscore(Yammss, Lu);
                                [~,~,Prec] = calcMultiClassScore(Yammss, Lu, num_class);
                                Time_res.(METHODS{method_i})(test_i) = temp_time;
                                ACC_res.(METHODS{method_i})(test_i) = ACC;
                                Fscore_res.(METHODS{method_i})(test_i) = F1;
                                Prec_res.(METHODS{method_i})(test_i) = Prec;
                            end
                            clear Yammss;


                        case 'AMGL'
                            cell_A = cell(num_views, 1);
                            for v = 1:num_views
                                Xl{v} = MD{v}(1:num_labeled, :);
                                Xu{v} = MD{v}(num_labeled+1:end, :);
                                cell_A{v} = constructW_PKN(MD{v}', k, 0);
                            end
                            tic
                            [Yamgl,~] = AMGL_Semi(cell_A, Ll);
                            temp_time = toc;
                            [ACC, F1] = accfscore(Yamgl, Lu);
                            [~,~,Prec] = calcMultiClassScore(Yamgl, Lu, num_class);
                            Time_res.(METHODS{method_i})(test_i) = temp_time;
                            ACC_res.(METHODS{method_i})(test_i) = ACC;
                            Fscore_res.(METHODS{method_i})(test_i) = F1;
                            Prec_res.(METHODS{method_i})(test_i) = Prec;
                            clear Yamgl;


                        case 'MLAN'
                            tic
                            [Ymlan,~] = myMLAN_SSC(MD, Ll, k);
                            temp_time = toc;
                            [ACC, F1] = accfscore(Ymlan, Lu);
                            [~,~,Prec] = calcMultiClassScore(Ymlan, Lu, num_class);
                            Time_res.(METHODS{method_i})(test_i) = temp_time;
                            ACC_res.(METHODS{method_i})(test_i) = ACC;
                            Fscore_res.(METHODS{method_i})(test_i) = F1;
                            Prec_res.(METHODS{method_i})(test_i) = Prec;
                            clear Ymlan;
                        
                        otherwise
                            error('Unknown method');
                    end
                end
            end

            %% save results
            save_path = ['./results/' ds_name '/mr', num2str(missing_rate*100), '_lr', num2str(labeled_rate*100) '/'];
            if exist(save_path) == 0
                mkdir(save_path);
            end
            for method_i = 1:length(METHODS)
                method_ACC = ACC_res.(METHODS{method_i});
                method_F1 = Fscore_res.(METHODS{method_i});
                method_Prec = Prec_res.(METHODS{method_i});
                method_Time = Time_res.(METHODS{method_i});
                save([save_path METHODS{method_i}, '_res.mat'], 'method_Time', 'method_ACC', ...
                    'method_F1', 'method_Prec');
            end
            clear X Y gnd;
        end
    end
end
