function result = segment_color(im_rgb, gd_thresh, se_size, gamma,min_edge)

     
    im_g = rgb2gray(im_rgb);

    % smooth the image into very distinct regions
    se = strel('disk', se_size); % structural element
        % opening by reconstruction
    im_e = imerode(im_g, se);
    im_obr = imreconstruct(im_e, im_g);
        % closing by reconstruction
    im_obr_d = imdilate(im_obr, se);
    im_ocbr = imreconstruct(imcomplement(im_obr_d), imcomplement(im_obr));
    im_ocbr = imcomplement(im_ocbr);

    % extract the boundaries of the regions
    gmag = imgradient(im_ocbr) > gd_thresh;
    % filtrate small edges
    gmag = bwareaopen(gmag, min_edge);

    % label the regions delimited by the boundaries
    [pieces_labeled, n_pieces] = bwlabel(1 - gmag);    

    % convert to LAB coordinates
    im = rgb2lab(im_rgb);
    
    % color each region with its mean value
    result = 0.5 * im .* gmag;
    for i=1 : n_pieces
        % region mask
        mask = pieces_labeled == i;
        % compute the mean color value of the region
        n_pixel = sum(mask(:));
        mean_color = sum(reshape(im .* mask, [], 3) ,1)/n_pixel;  % calculating mean color
        % adjust the color
        [~, j] = max(mean_color); % finding the most dominant color, j is the index
        mean_color(j) = mean_color(j).^gamma;  % gamma correction

        result = result + repmat(mask, 1, 1, 3) .* reshape(mean_color, 1,1,3);
    end
    
    
end

