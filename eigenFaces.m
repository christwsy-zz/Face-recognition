function [eigFaces, topN] = eigenFaces(train_imgs, pc_ratio)
    % prepare data for pca
    train_mean = mean(train_imgs); % overall mean for all training data
    recentered_train = train_imgs - repmat(train_mean, size(train_imgs, 1), 1);
    cov_train = 1 / size(recentered_train, 1) * recentered_train * recentered_train';
    [coeff, score, latent] = pca(cov_train);
    topN = round(length(latent) * pc_ratio); % choose some pc
    topEigVec = coeff(:,1:topN);
    eigFaces = recentered_train' * topEigVec; % eigenfaces
end
