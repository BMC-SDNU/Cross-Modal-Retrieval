function [W_image, W_text] = update_W_JFSSL(W_image, W_text, X_image, X_text, Y,...
    L_intra_image, L_intra_text, L_inter)
%Update W_image and W_text.
%Return new W_image and W_text.

%Initialize  auxiliary matrix R_image and R_text.
R_image = zeros(128, 128);
R_text = zeros(10, 10);

small_value = 0.00001;
temp = sum(W_image.^2, 2);
for num = 1:128
    R_image(num, num) = 1./ (2 .* sqrt(temp(num, 1) + small_value));
end

temp = sum(W_text.^2, 2);
for num = 1:10
    R_text(num,num) = 1./ (2 .* sqrt(temp(num,1) + small_value));
end
%lambda1 and lambda2 can produce much influence with different values.
lambda1 = 2.0;
lambda2 = 1.0;
temp_image = W_image;
temp_text = W_text;
%Update W_image
W_image = (X_image * X_image' + lambda1 .* R_image + ...
    lambda2 .* X_image * L_intra_image * X_image')^-1 * (X_image * Y...
    - lambda2 .* X_image * L_inter * X_text' * temp_text);
%Update W_text
W_text = (X_text * X_text' + lambda1 .* R_text + ...
    lambda2 .* X_text * L_intra_text * X_text')^-1 * (X_text * Y...
    - lambda2 .* X_text * L_inter * X_image' * temp_image);
