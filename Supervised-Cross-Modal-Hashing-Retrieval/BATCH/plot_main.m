close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

db = {'WIKI','MIRFLICKR','NUSWIDE10'};
loopnbits = [8 16 24 32 64 96 128];

param.top_K = 2000;


for dbi = 1   :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    
    % load dataset
    load(['./results/final_', db_name,'_result.mat']);
    
    hashmethods = {'CCQ','CMFH','CRE','FSH','LSSH','SCM-seq','SCRATCH','SePH-km','SMFH','SRSH','BATCH'};
    clear Image_VS_Text_MAP Text_VS_Image_MAP Image_VS_Text_recall Image_VS_Text_precision trainT...
        Text_VS_Image_recall Text_VS_Image_precision Image_To_Text_Precision Text_To_Image_Precision

    
    for ii = 1:length(loopnbits)
        for jj = 1:length(hashmethods)
            % MAP
            Image_VS_Text_MAP{jj,ii} = eva_info{jj,ii}.Image_VS_Text_MAP;
            Text_VS_Image_MAP{jj,ii} = eva_info{jj,ii}.Text_VS_Image_MAP;

            % Precision VS Recall
            Image_VS_Text_recall{jj,ii,:}    = eva_info{jj,ii}.Image_VS_Text_recall';
            Image_VS_Text_precision{jj,ii,:} = eva_info{jj,ii}.Image_VS_Text_precision';
            Text_VS_Image_recall{jj,ii,:}    = eva_info{jj,ii}.Text_VS_Image_recall';
            Text_VS_Image_precision{jj,ii,:} = eva_info{jj,ii}.Text_VS_Image_precision';

            % Top number Precision
            Image_To_Text_Precision{jj,ii,:} = eva_info{jj,ii}.Image_To_Text_Precision;
            Text_To_Image_Precision{jj,ii,:} = eva_info{jj,ii}.Text_To_Image_Precision;
            
			% Time
            trainT{jj,ii} = eva_info{jj,ii}.trainT;
        end
    end
    
    % plot attribution
    line_width=1;
    gcaline_width = 0.5;
    marker_size=5;
    xy_font_size=14;
    legend_font_size=12;
    title_font_size=14;
    gca_size = 14;
    location = 'northeast'; %'best'
    
    % save result
    result_URL = sprintf('./results/fig/%s/',db_name);
    if ~isdir(result_URL)
        mkdir(result_URL);
    end

    %% show mAP. This mAP function is provided by Yunchao Gong
    Image-to-Text
    figure('Color', [1 1 1]); hold on;
    for j = 1: length(hashmethods)
        Image_VS_Text_map = [];
        for i = 1: length(loopnbits)
            Image_VS_Text_map = [Image_VS_Text_map, Image_VS_Text_MAP{j, i}];
        end
        p = plot(1:length(loopnbits), Image_VS_Text_map);
        color=gen_color(j);
        marker=gen_marker(j);
        set(p,'Color', color);
        set(p,'Marker', marker);
        set(p,'LineWidth', line_width);
        set(p,'MarkerSize', marker_size);
    end
    set(gcf,'unit','centimeters','position',[10 5 16 14]);
    set(gca,'Position',[.2 .2 .65 .65],'fontsize',gca_size);

    h1 = xlabel('Number of bits');
    h2 = ylabel('mean Average Precision (mAP)');
    title('Image-to-Text','FontSize', title_font_size);
    set(h1, 'FontSize', xy_font_size);
    set(h2, 'FontSize', xy_font_size);
    set(gca, 'xtick', 1:length(loopnbits));
    set(gca, 'XtickLabel', cellstr(num2str(loopnbits(:)))');
    set(gca, 'linewidth', gcaline_width);
    hleg = legend(hashmethods);
    set(hleg, 'FontSize', legend_font_size);
    set(hleg, 'Location', location);
    %set(hleg,'Orientation','horizon') % extra legend
    set(gcf,'paperpositionmode','auto');
    box on;
    grid on;
    hold off;
    saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_mAP_ItoT',db_name,db_name),'fig');
    saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_mAP_ItoT',db_name,db_name),'jpg');
    saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_mAP_ItoT',db_name,db_name),'epsc2');    
    
    % Text-To-Image
    figure('Color', [1 1 1]); hold on;
    for j = 1: length(hashmethods)
        Text_VS_Image_map = [];
        for i = 1: length(loopnbits)
            Text_VS_Image_map = [Text_VS_Image_map, Text_VS_Image_MAP{j, i}];
        end
        p = plot(1:length(loopnbits), Text_VS_Image_map);
        color=gen_color(j);
        marker=gen_marker(j);
        set(p,'Color', color);
        set(p,'Marker', marker);
        set(p,'LineWidth', line_width);
        set(p,'MarkerSize', marker_size);
    end
    set(gcf,'unit','centimeters','position',[10 5 16 14]);
    set(gca,'Position',[.2 .2 .65 .65],'fontsize',gca_size);

    h1 = xlabel('Number of bits');
    h2 = ylabel('mean Average Precision (mAP)');
    title('Text-to-Image', 'FontSize', title_font_size);
    set(h1, 'FontSize', xy_font_size);
    set(h2, 'FontSize', xy_font_size);
    set(gca, 'xtick', 1:length(loopnbits));
    set(gca, 'XtickLabel', cellstr(num2str(loopnbits(:)))');
    set(gca, 'linewidth', gcaline_width);
    hleg = legend(hashmethods);
    set(hleg, 'FontSize', legend_font_size);
    set(hleg, 'Location', location);
    set(gcf,'paperpositionmode','auto');
    box on;
    grid on;
    hold off;
    saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_mAP_TtoI',db_name,db_name),'fig');
    saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_mAP_TtoI',db_name,db_name),'jpg');
    saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_mAP_TtoI',db_name,db_name),'epsc2');
    
    
    %% show precision vs recall
    for i = 1:length(loopnbits)
        str_nbits =  num2str(loopnbits(i));
        
        % Image-To-Text
        figure('Color', [1 1 1]); hold on;
        ind = [1:50:1000,1000];
        for j = 1: length(hashmethods)
            p = plot(Image_VS_Text_recall{j,i,:}(ind),Image_VS_Text_precision{j,i,:}(ind));
            color=gen_color(j);
            marker=gen_marker(j);
            set(p,'Color', color)
            set(p,'Marker', marker);
            set(p,'LineWidth', line_width);
            set(p,'MarkerSize', marker_size);
        end
        set(gcf,'unit','centimeters','position',[10 5 16 14]);
        set(gca,'Position',[.2 .2 .65 .65],'fontsize',gca_size);

        h1 = xlabel('Recall');
        h2 = ylabel('Precision');
        title(['Image-to-Text @ ',str_nbits,'-bits']);
        set(h1, 'FontSize', xy_font_size);
        set(h2, 'FontSize', xy_font_size);
        % hleg = legend(hashmethods);
        % set(hleg, 'FontSize', legend_font_size);
        % set(hleg,'Location', location);
        set(gca,'yTickLabel',num2str(get(gca,'yTick')','%.2f'))
        set(gca, 'linewidth', gcaline_width);
        set(gcf,'paperpositionmode','auto');
        box on;
        grid on;
        hold off;
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PR_ItoT@%s',db_name,db_name,str_nbits),'fig');
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PR_ItoT@%s',db_name,db_name,str_nbits),'jpg');
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PR_ItoT@%s',db_name,db_name,str_nbits),'epsc2');
        
        % Text-To-Image
        figure('Color', [1 1 1]); hold on;
        for j = 1: length(hashmethods)
            p = plot(Text_VS_Image_recall{j,i,:}(ind),Text_VS_Image_precision{j,i,:}(ind));
            color=gen_color(j);
            marker=gen_marker(j);
            set(p,'Color', color)
            set(p,'Marker', marker);
            set(p,'LineWidth', line_width);
            set(p,'MarkerSize', marker_size);
        end
        set(gcf,'unit','centimeters','position',[10 5 16 14]);
        set(gca,'Position',[.2 .2 .65 .65],'fontsize',gca_size);

        h1 = xlabel('Recall');
        h2 = ylabel('Precision');
        title(['Text-to-Image @ ',str_nbits,'-bits']);
        set(h1, 'FontSize', xy_font_size);
        set(h2, 'FontSize', xy_font_size);
        % hleg = legend(hashmethods);
        % set(hleg, 'FontSize', legend_font_size);
        % set(hleg,'Location', location);
        set(gca,'yTickLabel',num2str(get(gca,'yTick')','%.2f'))
        set(gca, 'linewidth', gcaline_width);
        set(gcf,'paperpositionmode','auto');
        box on;
        grid on;
        hold off;
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PR_TtoI@%s',db_name,db_name,str_nbits),'fig');
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PR_TtoI@%s',db_name,db_name,str_nbits),'jpg');
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PR_TtoI@%s',db_name,db_name,str_nbits),'epsc2');
    end
        
    %% Top number precision
    for i = 1:length(loopnbits)
        str_nbits =  num2str(loopnbits(i));
        
        % Image-To-Text
        figure('Color', [1 1 1]); hold on;
        pos = [1:50:1000,1000];
        for j = 1: length(hashmethods)
            p = plot(pos, Image_To_Text_Precision{j,i,:}(pos));
            color = gen_color(j);
            marker = gen_marker(j);
            set(p,'Color', color)
            set(p,'Marker', marker);
            set(p,'LineWidth', line_width);
            set(p,'MarkerSize', marker_size);
        end
        set(gcf,'unit','centimeters','position',[10 5 16 14]);
        set(gca,'Position',[.2 .2 .65 .65],'fontsize',gca_size);

        h1 = xlabel('N');
        h2 = ylabel('Precision');
        title(['Image-to-Text @ ',str_nbits,'-bits']);  
        set(h1, 'FontSize', xy_font_size);
        set(h2, 'FontSize', xy_font_size);
        % hleg = legend(hashmethods);
        % set(hleg, 'FontSize', legend_font_size);
        % set(hleg,'Location', location);
        set(gca,'yTickLabel',num2str(get(gca,'yTick')','%.2f'))
        set(gca, 'linewidth', gcaline_width);
        set(gcf,'paperpositionmode','auto');
        box on;
        grid on;
        hold off;
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PN_ItoT@%s',db_name,db_name,str_nbits),'fig');
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PN_ItoT@%s',db_name,db_name,str_nbits),'jpg');
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PN_ItoT@%s',db_name,db_name,str_nbits),'epsc2');
        
        % Text-To-Image
        figure('Color', [1 1 1]); hold on;
        for j = 1: length(hashmethods)
            p = plot(pos, Text_To_Image_Precision{j,i,:}(pos));
            color = gen_color(j);
            marker = gen_marker(j);
            set(p,'Color', color)
            set(p,'Marker', marker);
            set(p,'LineWidth', line_width);
            set(p,'MarkerSize', marker_size);
        end
        set(gcf,'unit','centimeters','position',[10 5 16 14]);
        set(gca,'Position',[.2 .2 .65 .65],'fontsize',gca_size);

        h1 = xlabel('N');
        h2 = ylabel('Precision');
        title(['Text-to-Image @ ',str_nbits,'-bits']);  
        set(h1, 'FontSize', xy_font_size);
        set(h2, 'FontSize', xy_font_size);
        % hleg = legend(hashmethods);
        % set(hleg, 'FontSize', legend_font_size);
        % set(hleg,'Location', location);
        set(gca,'yTickLabel',num2str(get(gca,'yTick')','%.2f'))
        set(gca, 'linewidth', gcaline_width);
        set(gcf,'paperpositionmode','auto');
        box on;
        grid on;
        hold off;
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PN_TtoI@%s',db_name,db_name,str_nbits),'fig');
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PN_TtoI@%s',db_name,db_name,str_nbits),'jpg');
        saveas(gcf, sprintf('./results/fig/%s/final_%s_fig_PN_TtoI@%s',db_name,db_name,str_nbits),'epsc2');
    end
        
end