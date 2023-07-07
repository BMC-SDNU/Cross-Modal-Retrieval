function [map] = map_at_50(sim_x,L_tr,L_te)

% written by Devraj Mandal on 2nd Nov 2016

%sim_x(i,j) denote the sim bewteen query j and database i
tn = size(sim_x,2);
APx = zeros(tn,1);
R = 50;

for i = 1 : tn
    Px = zeros(R,1);
    deltax = zeros(R,1);
    label = L_te(i,:);
    [~,inxx] = sort(sim_x(:,i),'descend');
    
    % compute Lx - the denominator in the map calculation
    % Lx = 1 if the retrieved item has the same label with the query or
    % shares atleast one label else Lx = 0
    search_set = L_tr(inxx(1:R),:);
        
    for r = 1 : R        
        
        
        Lrx = sum(diag(repmat(label,r,1)*search_set(1:r,:).')>0);
        
%         Lrx = 0;
%         for j=1:r
%             if sum(label*(search_set(j,:)).')>0
%                 Lrx = Lrx+1;
%             end
%         end        
        
        if sum(label*(search_set(r,:)).')>0
            deltax(r) = 1;
        end
        
        Px(r) = Lrx/r;
    end
    Lx = sum(deltax);
    if Lx ~=0
        APx(i) = sum(Px.*deltax)/Lx;
    end
end
map = mean(APx);