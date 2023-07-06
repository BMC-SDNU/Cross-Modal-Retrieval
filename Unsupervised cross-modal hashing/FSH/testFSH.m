clear all; warning off; clc;
load uci_all;
%%
I_te = normalize1(I_te); I_tr = normalize1(I_tr);
T_te = normalize1(T_te); T_tr = normalize1(T_tr);
I_te = bsxfun(@minus, I_te, mean(I_tr, 1));
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_te = bsxfun(@minus, T_te, mean(T_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));
%%
bit = 64;
opt.k = 10;
opt.Nsamp = 50;
opt.iter = 100;
opt.lambda = 300;
opt.lam =  0.5;
opt.cca = 0;
opt.km = 1;
[Wx, Wy, ~] = trainFSH(I_tr', T_tr', bit, opt);
%%
B1 = sign(real(I_tr * Wx));
B2 = sign(real(T_tr * Wy));
tB1 = sign(real(I_te * Wx));
tB2 = sign(real(T_te * Wy));
%%
sim_it = B1 * tB2'; sim_ti = B2 * tB1';
map1 = mAP(sim_it,L_tr,L_te,0)
map2 = mAP(sim_ti,L_tr,L_te,0)
