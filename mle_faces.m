function accu = mle_faces(train_imgs, test_imgs, num_subjects, w, h)
    % training, prepare useful data
    train_class = train_imgs(:,1);
    test_class = test_imgs(:,1);
    train_imgs_no_class = train_imgs(:, 2:end);
	test_imgs_no_feat = test_imgs(:, 2:end);
    train_imgs_no_class = double(train_imgs_no_class);
	test_imgs_no_feat = double(test_imgs_no_feat);

	% number of image for each class in training and testing sets
	num_per_class_train = size(train_imgs_no_class, 1) / num_subjects;
	num_per_class_test = size(test_imgs, 1) / num_subjects;
	
	% pca
    [eigFaces, topN] = eigenFaces(train_imgs_no_class, 0.1);
	
    % train_proj will be the new original data
    train_proj_no_class = train_imgs_no_class * eigFaces; % num_of_train x num_of_pc
	test_proj_no_class = test_imgs_no_feat * eigFaces;
	
	means = zeros(num_subjects, size(train_proj_no_class, 2));
	covs = zeros(num_subjects, size(train_proj_no_class, 2) ^ 2);
	for i = 1:num_subjects
		start_idx = (i - 1) * num_per_class_train + 1;
		end_idx = start_idx + num_per_class_train - 1;
		class_data = train_proj_no_class(start_idx: end_idx, :);
		means(i,:) = mean(class_data);
		covs(i,:) = reshape(cov(class_data), 1, size(train_proj_no_class, 2) ^ 2);
	end
% 	disp(size(means));
% 	disp(size(covs));
	
	% testing
	correct = 0;
	wrong = 0;
	
	% assume that all the classes have the same prior
	for i = 1:size(test_proj_no_class, 1)
		n = size(means, 1);
		recentered = repmat(test_proj_no_class(i,:), n, 1) - means;
		
		
	end
		
	accu = correct / (correct + wrong);
end

function l = lklhood(data, cov_mat)
    A = -data/2;
    first = (A / cov_mat) * (data');
    l = exp(first)./sqrt(det(cov_mat));
end
