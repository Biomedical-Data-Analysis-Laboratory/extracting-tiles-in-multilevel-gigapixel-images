from skimage.measure import regionprops
from skimage import morphology
from scipy import ndimage
import numpy as np
import pyvips
import utils
import time
import cv2
import os

# Start timer
current_start_time = time.time()

# In the WSI folder, there is a file containing a dict with the 7 binary masks.
# To specify which of these masks to use, list the tissue types in the following variable.
# Available options: ['background', 'blood', 'damaged', 'muscle', 'stroma', 'urothelium']
tissue_classes_to_fit_tiles_on = ['urothelium']

# How large percentage of the tile must cover the region to be consider a valid tile.
# Float between 0-1.
PHI = 0.8

# Which level in the WSI to use when checking for valid tiles.
# Available options: '25x', '100x', '400x'.
ALPHA = '25x'

# All valid tiles are displayed on the 25x image and saved as JPEG to the folder.
# This option determines which of the three levels to include in the final image.
# Tiles from all three levels in the WSI are saved, this option is only for visualization.
# Available options: ['25x', '100x', '400x'].
TILES_TO_SHOW = ['25x', '100x', '400x']

# Size of width/height of the tiles to extract. Integer.
TILE_SIZE = 256

# The level the annotation mask is on, and also the level our tiles are on. For our mask, that's '25x'.
TAU = '25x'

# The binary masks contain small regions which is not of interest.
# These are removed using the remove_small_objects() function.
# This variable sets the minimum size to remove.
# Available options: Integer values, usually between 500 and 20000.
SMALL_REGION_REMOVE_THRESHOLD = 3000

# Paths
wsi_dataset_file_path = 'WSIs/'
output_folder = 'Output/'
os.makedirs(output_folder, exist_ok=True)

# Variable initialization
dict_of_all_predicted_coordinates_dict = dict()
list_of_tiles_from_current_wsi = []
run_once = 0

region_masks = dict()
ratio_dict = dict()
ratio_dict['400x'] = 1
ratio_dict['100x'] = 4
ratio_dict['25x'] = 16

# Loop through each WSI in the 'WSIs/' folder
for wsi_name in os.listdir(wsi_dataset_file_path):

    # Load Colormap pickle file for urothelium
    colormap_image_all_classes_list = utils.pickle_load(wsi_dataset_file_path + wsi_name + '/' + 'COLORMAP_IMAGES_PICKLE.obj')

    # Read images
    full_image_400 = pyvips.Image.new_from_file(wsi_dataset_file_path + wsi_name + '/' + wsi_name + '.scn', level=0).flatten().rot(1)
    full_image_100 = pyvips.Image.new_from_file(wsi_dataset_file_path + wsi_name + '/' + wsi_name + '.scn', level=1).flatten().rot(1)
    full_image_25 = pyvips.Image.new_from_file(wsi_dataset_file_path + wsi_name + '/' + wsi_name + '.scn', level=2).flatten().rot(1)

    # Process SCN image
    scn_offset_400x_x, scn_offset_400x_y, _, _ = utils.remove_white_background_v3(input_img=full_image_400, PADDING=0)
    scn_offset_100x_x, scn_offset_100x_y, _, _ = utils.remove_white_background_v3(input_img=full_image_100, PADDING=0)
    scn_offset_25x_x, scn_offset_25x_y, scn_width_25x, scn_height_25x = utils.remove_white_background_v3(input_img=full_image_25, PADDING=0)

    # Add all offsets in a dict
    offset_dict_x = dict()
    offset_dict_x['25x'] = scn_offset_25x_x
    offset_dict_x['100x'] = scn_offset_100x_x
    offset_dict_x['400x'] = scn_offset_400x_x
    offset_dict_y = dict()
    offset_dict_y['25x'] = scn_offset_25x_y
    offset_dict_y['100x'] = scn_offset_100x_y
    offset_dict_y['400x'] = scn_offset_400x_y

    # Loop through each tissue class to fit tiles on
    for current_class_to_copy in tissue_classes_to_fit_tiles_on:

        print('Now processing {} regions'.format(current_class_to_copy))

        # Extract tissue mask for current class
        current_class_colormap = colormap_image_all_classes_list[utils.tissue_class_to_index[current_class_to_copy]].copy()

        # Resize colormap to the size of 25x overview image
        current_class_colormap = cv2.resize(current_class_colormap, dsize=(scn_width_25x, scn_height_25x), interpolation=cv2.INTER_CUBIC)
        print('Loaded segmentation mask with size {} x {}'.format(current_class_colormap.shape[1], current_class_colormap.shape[0]))

        # Uncomment next two lines if you want to save the annotation mask image
        # annotation_mask_for_saving = current_class_colormap * 255
        # cv2.imwrite(output_folder + 'Annotation_mask_{}.jpg'.format(current_class_to_copy), annotation_mask_for_saving, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Use a boolean condition to find where pixel values are > 0.75
        blobs = current_class_colormap > 0.75

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
        region_masks[current_class_to_copy] = np.zeros(shape=(current_class_colormap.shape[0], current_class_colormap.shape[1]))

        # Extract all coordinates (to draw region on overview image)
        for current_region in list_of_regions:
            for current_region_coordinate in current_region.coords:
                region_masks[current_class_to_copy][current_region_coordinate[0], current_region_coordinate[1]] = 1

        # Create a grid of all possible x- and y-coordinates
        all_x_pos, all_y_pos = [], []
        for x_pos in range(0, int(current_class_colormap.shape[1] - TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))):
            all_x_pos.append(x_pos)
        for y_pos in range(0, int(current_class_colormap.shape[0] - TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))):
            all_y_pos.append(y_pos)

        # Create a new list with all xy-positions in current SCN image
        list_of_tiles_from_current_region = []
        for y_pos in all_y_pos:
            for x_pos in all_x_pos:
                # Equation 2 in paper
                if int(sum(sum(region_masks[current_class_to_copy][y_pos:y_pos + int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])),
                               x_pos:x_pos + int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))]))) >= (pow((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), 2) * PHI):
                    list_of_tiles_from_current_region.append((x_pos, y_pos))

        # Add the tiles to the list of tiles of current wsi
        list_of_tiles_from_current_wsi.extend(list_of_tiles_from_current_region)
        tile_x = dict()
        tile_y = dict()

        # Save coordinates for each tiles to a dict to create a dataset
        for current_xy_pos in list_of_tiles_from_current_region:

            # Equation 3 in paper.
            for BETA in ['25x', '100x', '400x']:
                # with offset
                tile_x[BETA] = offset_dict_x[BETA] + (current_xy_pos[0] + (TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2) * (ratio_dict[TAU] / ratio_dict[BETA]) - TILE_SIZE / 2
                tile_y[BETA] = offset_dict_y[BETA] + (current_xy_pos[1] + (TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2) * (ratio_dict[TAU] / ratio_dict[BETA]) - TILE_SIZE / 2

            # Save tile to coordinate dict (coordinate of top-left corner, with white offset)
            id_number = len(dict_of_all_predicted_coordinates_dict.keys())
            dict_of_all_predicted_coordinates_dict[id_number] = dict()
            dict_of_all_predicted_coordinates_dict[id_number]['path'] = wsi_dataset_file_path + wsi_name + '/' + wsi_name + '.scn'
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_400x'] = (int(tile_x['400x']), int(tile_y['400x']))
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_100x'] = (int(tile_x['100x']), int(tile_y['100x']))
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_25x'] = (int(tile_x['25x']), int(tile_y['25x']))
            dict_of_all_predicted_coordinates_dict[id_number]['tissue_type'] = current_class_to_copy

    # Save predicted coordinates dict as pickle
    utils.pickle_save(dict_of_all_predicted_coordinates_dict, output_folder + 'coordinate_tissue_predictions_pickle.obj')

    # Extract overview image from 25x WSI, and save as jpeg.
    overview_img = full_image_25.extract_area(scn_offset_25x_x, scn_offset_25x_y, scn_width_25x, scn_height_25x)

    # Save image
    filename = output_folder + 'image_clean.jpeg'
    overview_img.jpegsave(filename, Q=100)

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

    print('Found {} tiles in image'.format(len(list_of_tiles_from_current_wsi)))

    # Draw tiles on the overview image
    for current_xy_pos in list_of_tiles_from_current_wsi:

        general_start_x = dict()
        general_start_y = dict()
        general_end_x = dict()
        general_end_y = dict()

        # Equation 4 in paper.
        # Note that we are drawing these tiles on the extracted 25x image, which means we do not need to add the offset-values.
        for BETA in ['25x', '100x', '400x']:
            general_start_x[BETA] = int(((current_xy_pos[0] + ((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2)) * (ratio_dict[TAU] / ratio_dict['25x'])) - ((TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x'])) / 2))
            general_start_y[BETA] = int(((current_xy_pos[1] + ((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2)) * (ratio_dict[TAU] / ratio_dict['25x'])) - ((TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x'])) / 2))
            general_end_x[BETA] = int(general_start_x[BETA] + TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x']))
            general_end_y[BETA] = int(general_start_y[BETA] + TILE_SIZE * (ratio_dict[BETA] / ratio_dict['25x']))

        # Draw tiles (Red tile indicate which level ALPHA is, and the other levels are shown in green)
        for draw_level in TILES_TO_SHOW:
            color = (0, 0, 255) if draw_level == ALPHA else (0, 255, 0)
            cv2.rectangle(overview_jpeg_file, (general_start_x[draw_level], general_start_y[draw_level]), (general_end_x[draw_level], general_end_y[draw_level]), color, 3)

    # Save overview image
    cv2.imwrite(output_folder + 'image_with_mask_and_tiles_alpha_{}_phi_{}_{}.jpg'.format(ALPHA, PHI, len(list_of_tiles_from_current_wsi)), overview_jpeg_file, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # Finished
    # Calculate elapse time for current run
    elapse_time = time.time() - current_start_time
    m, s = divmod(elapse_time, 60)
    h, m = divmod(m, 60)
    model_time = '%02d:%02d:%02d' % (h, m, s)

    # Print out results
    print('Finished. Duration: {}'.format(model_time))
