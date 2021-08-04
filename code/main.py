from skimage.measure import regionprops
from skimage import morphology
from scipy import ndimage
import numpy as np
import pyvips
import pickle
import time
import cv2
import os

# Set this option to True to save tiles as JPEG images in a separate folder.
save_tiles_as_jpeg = False

# Set this option to True to save the binary annotation mask as a JPEG image.
save_binary_annotation_mask = False

# In the WSI folder, there is a file containing a dict with the 7 binary masks.
# To specify which of these masks to use, list the tissue types in the following variable.
# Available options: ['urothelium', 'stroma', 'muscle', 'blood', 'damaged', 'background']
tissue_classes_to_fit_tiles_on = ['urothelium']

# How large percentage of the tile must cover the region to be consider a valid tile.
# Float between 0-1.
PHI = 0.7

# Which level in the WSI to use when checking for valid tiles (level 0=400x, level 1=100x, and level 2=25x).
# Available options: '25x', '100x', '400x'.
ALPHA = '400x'

# All valid tiles are displayed on the 25x image and saved as JPEG to the folder.
# This option determines which of the three levels to include in the final image.
# Tiles from all three levels in the WSI are saved, this option is only for visualization.
# Available options: ['25x', '100x', '400x'].
TILES_TO_SHOW = ['400x']

# Size of width/height of the tiles to extract. Integer.
TILE_SIZE = 256

# The level the annotation mask is on, and also the level our tiles are on. For our mask it's '25x'.
TAU = '25x'

# The binary masks contain small regions which is not of interest.
# These are removed using the remove_small_objects() function.
# This variable sets the minimum size to remove.
# Available options: Integer values, usually between 500 and 20000.
SMALL_REGION_REMOVE_THRESHOLD = 3000

# Paths
wsi_dataset_file_path = 'WSIs/'
output_folder = 'Output/'
extracted_tiles_folder = 'Extracted_tiles/'
os.makedirs(output_folder, exist_ok=True)
if save_tiles_as_jpeg:
    os.makedirs(extracted_tiles_folder, exist_ok=True)

# Variable initialization
dict_of_all_predicted_coordinates_dict = dict()
list_of_valid_tiles_from_current_wsi = []
i = 0

# Create a dict containing the ratio for each level
region_masks = dict()
ratio_dict = dict()
ratio_dict['400x'] = 1
ratio_dict['100x'] = 4
ratio_dict['25x'] = 16

# Create a dict containin the index of each class
tissue_class_to_index = dict()
tissue_class_to_index['background'] = 0
tissue_class_to_index['blood'] = 1
tissue_class_to_index['damaged'] = 2
tissue_class_to_index['muscle'] = 3
tissue_class_to_index['stroma'] = 4
tissue_class_to_index['urothelium'] = 5
tissue_class_to_index['undefined'] = 6

# Loop through each WSI
for wsi_name in os.listdir(wsi_dataset_file_path):

    # Start timer
    current_wsi_start_time = time.time()

    # Create folder for each WSI to store output
    os.makedirs(output_folder + wsi_name, exist_ok=True)
    if save_tiles_as_jpeg:
        os.makedirs(extracted_tiles_folder + wsi_name, exist_ok=True)

    # Load annotation mask. For us, this is a pickle file containing the annotation mask for all tissue classes.
    annotation_mask_path = wsi_dataset_file_path + wsi_name + '/' + 'COLORMAP_IMAGES_PICKLE.obj'
    with open(annotation_mask_path, 'rb') as handle:
        annotation_mask_all_classes = pickle.load(handle)

    # Read images
    full_image_400 = pyvips.Image.new_from_file(wsi_dataset_file_path + wsi_name + '/' + wsi_name + '.scn', level=0, autocrop=True).flatten().rot(1)
    full_image_100 = pyvips.Image.new_from_file(wsi_dataset_file_path + wsi_name + '/' + wsi_name + '.scn', level=1, autocrop=True).flatten().rot(1)
    full_image_25 = pyvips.Image.new_from_file(wsi_dataset_file_path + wsi_name + '/' + wsi_name + '.scn', level=2, autocrop=True).flatten().rot(1)

    # Find width/heigh of 25x image
    scn_width_25x = full_image_25.width
    scn_height_25x = full_image_25.height

    # Loop through each tissue class to fit tiles on
    for current_class_to_copy in tissue_classes_to_fit_tiles_on:
        print('Now processing {} regions'.format(current_class_to_copy))

        # Extract mask for current class
        current_class_mask = annotation_mask_all_classes[tissue_class_to_index[current_class_to_copy]].copy()

        # Resize colormap to the size of 25x overview image
        current_class_mask = cv2.resize(current_class_mask, dsize=(scn_width_25x, scn_height_25x), interpolation=cv2.INTER_CUBIC)
        print('Loaded segmentation mask with size {} x {}'.format(current_class_mask.shape[1], current_class_mask.shape[0]))

        # Save the annotation mask image (If option is set to True)
        if save_binary_annotation_mask:
            annotation_mask_for_saving = current_class_mask * 255
            cv2.imwrite(output_folder + wsi_name + '/Binary_annotation_mask_{}.jpg'.format(current_class_to_copy), annotation_mask_for_saving, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Use a boolean condition to find where pixel values are > 0.75
        blobs = current_class_mask > 0.75

        # label connected regions that satisfy this condition
        labels, regions_found_in_wsi_before_removing_small_obj = ndimage.label(blobs)
        print('\tFound {} regions'.format(regions_found_in_wsi_before_removing_small_obj))

        # Remove all the small regions
        labels = morphology.remove_small_objects(labels, min_size=SMALL_REGION_REMOVE_THRESHOLD)

        # Get region properties
        list_of_regions = regionprops(labels)

        n_regions_after_removing_small_obj = len(list_of_regions)
        print('\tFound {} regions (after removing small objects)'.format(n_regions_after_removing_small_obj))

        # Create a new binary map after removing small objects
        region_masks[current_class_to_copy] = np.zeros(shape=(current_class_mask.shape[0], current_class_mask.shape[1]))

        # Extract all coordinates (to draw region on overview image)
        for current_region in list_of_regions:
            for current_region_coordinate in current_region.coords:
                region_masks[current_class_to_copy][current_region_coordinate[0], current_region_coordinate[1]] = 1

        # Create a grid of all possible x- and y-coordinates (starting position (0,0))
        all_x_pos, all_y_pos = [], []
        for x_pos in range(0, int(current_class_mask.shape[1] - TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))):
            all_x_pos.append(x_pos)
        for y_pos in range(0, int(current_class_mask.shape[0] - TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))):
            all_y_pos.append(y_pos)

        # Create a new list with all xy-positions in current SCN image
        list_of_valid_tiles_from_current_class = []
        for y_pos in all_y_pos:
            for x_pos in all_x_pos:
                # Equation 1 in paper
                if int(sum(sum(region_masks[current_class_to_copy][y_pos:y_pos + int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])),
                               x_pos:x_pos + int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))]))) >= (pow((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), 2) * PHI):
                    list_of_valid_tiles_from_current_class.append((x_pos, y_pos))

        # Add the tiles to the list of tiles of current wsi
        list_of_valid_tiles_from_current_wsi.extend(list_of_valid_tiles_from_current_class)
        tile_x = dict()
        tile_y = dict()

        # Save coordinates for each tiles to a dict to create a dataset
        for current_xy_pos in list_of_valid_tiles_from_current_wsi:

            # Equation 2 in paper.
            for BETA in ['25x', '100x', '400x']:
                tile_x[BETA] = (current_xy_pos[0] + (TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2) * (ratio_dict[TAU] / ratio_dict[BETA]) - TILE_SIZE / 2
                tile_y[BETA] = (current_xy_pos[1] + (TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2) * (ratio_dict[TAU] / ratio_dict[BETA]) - TILE_SIZE / 2

            # Save tile to coordinate dict (coordinate of top-left corner)
            id_number = len(dict_of_all_predicted_coordinates_dict.keys())
            dict_of_all_predicted_coordinates_dict[id_number] = dict()
            dict_of_all_predicted_coordinates_dict[id_number]['path'] = wsi_dataset_file_path + wsi_name + '/' + wsi_name + '.scn'
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_400x'] = (int(tile_x['400x']), int(tile_y['400x']))
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_100x'] = (int(tile_x['100x']), int(tile_y['100x']))
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_25x'] = (int(tile_x['25x']), int(tile_y['25x']))
            dict_of_all_predicted_coordinates_dict[id_number]['tissue_type'] = current_class_to_copy

            # Extract and save tiles as jpeg-images (If option is set to True)
            if save_tiles_as_jpeg:
                tile_400x = full_image_400.extract_area(int(tile_x['400x']), int(tile_y['400x']), TILE_SIZE, TILE_SIZE)
                tile_100x = full_image_100.extract_area(int(tile_x['100x']), int(tile_y['100x']), TILE_SIZE, TILE_SIZE)
                tile_25x = full_image_25.extract_area(int(tile_x['25x']), int(tile_y['25x']), TILE_SIZE, TILE_SIZE)
                tile_400x.jpegsave(extracted_tiles_folder + wsi_name + '/tile_{}_400x.jpeg'.format(i), Q=100)
                tile_100x.jpegsave(extracted_tiles_folder + wsi_name +'/tile_{}_100x.jpeg'.format(i), Q=100)
                tile_25x.jpegsave(extracted_tiles_folder + wsi_name + '/tile_{}_25x.jpeg'.format(i), Q=100)
                i += 1

    # Save predicted coordinates dict as pickle
    with open(output_folder + wsi_name + '/coordinate_tissue_predictions_pickle.obj', 'wb') as handle:
        pickle.dump(dict_of_all_predicted_coordinates_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save overview image
    filename = output_folder + wsi_name + '/image_clean.jpeg'
    full_image_25.jpegsave(filename, Q=100)

    # Read overview image again using cv2, and add alpha channel to overview image.
    overview_jpeg_file = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    overview_jpeg_file = np.dstack([overview_jpeg_file, np.ones((overview_jpeg_file.shape[0], overview_jpeg_file.shape[1]), dtype="uint8") * 255])

    # Convert masks from 0-1 -> 0-255 (can also be used to set the color)
    for n in tissue_classes_to_fit_tiles_on:
        region_masks[n] *= 255

    # Resize masks to same size as the overview image
    for n in tissue_classes_to_fit_tiles_on:
        region_masks[n] = cv2.resize(region_masks[n], dsize=(overview_jpeg_file.shape[1], overview_jpeg_file.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Create a empty alpha channel
    alpha_channel = np.zeros(shape=(region_masks[tissue_classes_to_fit_tiles_on[0]].shape[0], region_masks[tissue_classes_to_fit_tiles_on[0]].shape[1]))

    # Each mask is 1-channel, merge them to create a 3-channel image (RGB), the order is used to set the color for each mask. Add a alpha-channel.
    if len(tissue_classes_to_fit_tiles_on) >= 1:
        region_masks[tissue_classes_to_fit_tiles_on[0]] = cv2.merge((region_masks[tissue_classes_to_fit_tiles_on[0]], alpha_channel, alpha_channel, alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 2:
        region_masks[tissue_classes_to_fit_tiles_on[1]] = cv2.merge((alpha_channel, region_masks[tissue_classes_to_fit_tiles_on[1]], alpha_channel, alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 3:
        region_masks[tissue_classes_to_fit_tiles_on[2]] = cv2.merge((alpha_channel, alpha_channel, region_masks[tissue_classes_to_fit_tiles_on[2]], alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 4:
        region_masks[tissue_classes_to_fit_tiles_on[3]] = cv2.merge((region_masks[tissue_classes_to_fit_tiles_on[3]], region_masks[tissue_classes_to_fit_tiles_on[3]], alpha_channel, alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 5:
        region_masks[tissue_classes_to_fit_tiles_on[4]] = cv2.merge((region_masks[tissue_classes_to_fit_tiles_on[4]], alpha_channel, region_masks[tissue_classes_to_fit_tiles_on[4]], alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 6:
        region_masks[tissue_classes_to_fit_tiles_on[5]] = cv2.merge((alpha_channel, region_masks[tissue_classes_to_fit_tiles_on[5]], region_masks[tissue_classes_to_fit_tiles_on[5]], alpha_channel))

    # Draw the selected regions on the overview image
    for _, current_tissue_mask in region_masks.items():
        overview_jpeg_file = cv2.addWeighted(current_tissue_mask, 1, overview_jpeg_file, 1.0, 0, dtype=cv2.CV_64F)

    # Draw tiles on the overview image
    for current_xy_pos in list_of_valid_tiles_from_current_wsi:
        start_x = dict()
        start_y = dict()
        end_x = dict()
        end_y = dict()

        # Equation 3 in paper.
        for BETA in ['25x', '100x', '400x']:
            start_x[BETA] = int(((current_xy_pos[0] + ((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2)) * (ratio_dict[TAU] / ratio_dict['25x'])) - ((TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x'])) / 2))
            start_y[BETA] = int(((current_xy_pos[1] + ((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2)) * (ratio_dict[TAU] / ratio_dict['25x'])) - ((TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x'])) / 2))
            end_x[BETA] = int(start_x[BETA] + TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x']))
            end_y[BETA] = int(start_y[BETA] + TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x']))

        # Draw tiles (Red tiles indicate which level ALPHA is, and the corresponding levels are shown in green)
        for draw_level in TILES_TO_SHOW:
            color = (0, 0, 255) if draw_level == ALPHA else (0, 255, 0)
            cv2.rectangle(overview_jpeg_file, (start_x[draw_level], start_y[draw_level]), (end_x[draw_level], end_y[draw_level]), color, 3)

    # Save overview image
    cv2.imwrite(output_folder + wsi_name + '/image_with_mask_and_tiles_alpha_{}_phi_{}_{}.jpg'.format(ALPHA, PHI, len(list_of_valid_tiles_from_current_wsi)), overview_jpeg_file, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # Calculate elapse time for current run
    elapse_time = time.time() - current_wsi_start_time
    m, s = divmod(elapse_time, 60)
    h, m = divmod(m, 60)
    model_time = '%02d:%02d:%02d' % (h, m, s)

    # Print out results
    print('Found {} tiles in image'.format(len(list_of_valid_tiles_from_current_wsi)))
    print('Finished. Duration: {}'.format(model_time))
