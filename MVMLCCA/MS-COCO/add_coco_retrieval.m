function [mAP12_array,mAP21_array] = add_coco_retrieval(Wx,D,I_val_ResNet101,T_val_Word2Vec_caption,Z_val_1,Z_val_2,ld,D_power)
[a, index] = sort(diag(D),'descend');
D = diag(a);
Wx = Wx(:,index);
% %D_power = 0;
D = D^D_power;
I_val_projected = full(MyNormalization(I_val_ResNet101))*Wx(1:size(I_val_ResNet101,2),:)*D;
T_val_projected = full(MyNormalization(T_val_Word2Vec_caption))*Wx(size(I_val_ResNet101,2)+1: size(I_val_ResNet101,2) + size(T_val_Word2Vec_caption,2),:)*D;

I_val_projected = I_val_projected(:,1:upto_ld);
T_val_projected = T_val_projected(:,1:upto_ld);


mAP12 =0;
mAP21 = 0;
[_mAP , _mAP21] = coco_common_retrieval(I_val_projected(:,1:ld),T_val_projected(:,1:ld),Z_val_1,Z_val_2,50);

mAP12 = _mAP
mAP21 = _mAP21
