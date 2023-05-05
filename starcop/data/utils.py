import fsspec
import glob
import numpy as np
import rasterio

def load_all_tile_indices_from_folder(settings_dataset):
    allbands = glob.glob(settings_dataset["data_base_path"]+"/*_allbands.tif")
    allbands.sort()

    if "FC_dataset_min200_padmin20" in settings_dataset.data_base_path:
        ignore_list = ["046", "034", "012"] # < for FC dataset
    if "PB_dataset_min200_padmin20" in settings_dataset.data_base_path:
        ignore_list = ["001484", "001916", "001917", "001918", "001919", "001920", "001921", "001922", "001923", "001924", "001925", "001926", "001927", "001928", "001929", "01916", "01917", "01918", "01919", "01920", "01921", "01922", "01923", "01924", "01925", "01926", "01927", "01928", "01929", "001484"] # < for PB dataset
        
    tiles = []

    for idx,allband_file in enumerate(allbands):
        # print(idx,allband_file)
        idx_desc = allband_file.split("/")[-1].split("_ang")[0]
        filename = allband_file.split("/")[-1].split("_allbands.tif")[0]
        if idx_desc in ignore_list:
            # print("skipping", allband_file)
            continue

        # print(idx_desc, filename)
        tiles_from_file = file_to_tiles_indices(filename, settings_dataset, 
            tile_px_size = settings_dataset.tile_px_size, tile_overlap_px = settings_dataset.tile_overlap_px, 
            include_last_row_colum_extra_tile = settings_dataset.include_last_row_colum_extra_tile)

        tiles += tiles_from_file


    print("Loaded:", len(tiles), "total tile indices")
    return tiles

    # print("We got", len(tiles), "tiles",tiles)
    # tilex,tiley = load_tile_idx(tiles[0])
    # print(tilex.shape,tiley.shape)

def file_to_tiles_indices(filename, settings, tile_px_size = 128, tile_overlap_px = 4, 
                          include_last_row_colum_extra_tile = True):
    """
    Opens one tif file and extracts all tiles (given tile size and overlap).
    Returns list of indices to the tile (to postpone in memory loading).
    """
 
    if settings["dataset_mode"] == "regression_output":  # < for Four Corners
        shape_file = settings["data_base_path"] + filename + settings["mask_file"]
    if settings["dataset_mode"] == "segmentation_output":  # < for Permian Basin
        shape_file = settings["data_base_path"] + filename + settings["label_file"]

    with rasterio.open(shape_file) as src:
        filename_shape = src.height, src.width

    data_h, data_w = filename_shape
    if data_h < tile_px_size or data_w < tile_px_size:
        # print("skipping, too small!")
        return []

    h_tiles_n = int(np.floor((data_h-tile_overlap_px) / (tile_px_size-tile_overlap_px)))
    w_tiles_n = int(np.floor((data_w-tile_overlap_px) / (tile_px_size-tile_overlap_px)))

    tiles = []
    tiles_X = []
    tiles_Y = []
    for h_idx in range(h_tiles_n):
            for w_idx in range(w_tiles_n):
                    tiles.append([w_idx * (tile_px_size-tile_overlap_px), h_idx * (tile_px_size-tile_overlap_px)])
    if include_last_row_colum_extra_tile:
            for w_idx in range(w_tiles_n):
                    tiles.append([w_idx * (tile_px_size-tile_overlap_px), data_h - tile_px_size])
            for h_idx in range(h_tiles_n):
                    tiles.append([data_w - tile_px_size, h_idx * (tile_px_size-tile_overlap_px)])
            tiles.append([data_w - tile_px_size, data_h - tile_px_size])

    # Save file ID + corresponding tiles[]
    tiles_indices = [[filename]+t+[tile_px_size,tile_px_size] for t in tiles]
    return tiles_indices

def select_bands(nm, ranges_of_interest):
    selected_bands = []
    for range in ranges_of_interest:
        # range = [2000,2510]
        a,b = range
        add_idx = list(np.nonzero((nm > a) & (nm < b))[0])
        print("adding from", nm[add_idx[0]], " to ", nm[add_idx[-1]])

        selected_bands += add_idx
    selected_bands = list(set(selected_bands))
    selected_bands.sort()

    return selected_bands


def load_tile_nanometers_descriptors(tile, settings):
    filename, x, y, w, h = tile
    # print("loading tile", tile)

    # load window:
    window = rasterio.windows.Window(row_off=y, col_off=x, width=w, height=h)
    allband_file = settings.data_base_path + filename + settings.allband_file
        
    with rasterio.open(allband_file) as src:
        # Select bands intelligently ...
        nanometers = np.asarray([float(d.replace(" Nanometers","")) for d in src.descriptions])
        bandlist = select_bands(nanometers, settings.bands.band_ranges)
        selected_nanometers = nanometers[bandlist]
    
    selected_nanometers = [int(n) for n in selected_nanometers] # enough to keep the int value
    print("in total selected", len(selected_nanometers), "bands to load")
    return bandlist, selected_nanometers


def load_tile_idx(tile, settings, bandlist, load_x = True):
    """
    Loads tile data values from the saved indices (file and window locations).
    """
    filename, x, y, w, h = tile
    # print("loading tile", tile)

    # load window:
    window = rasterio.windows.Window(row_off=y, col_off=x, width=w, height=h)

    allband_file = settings.data_base_path + filename + settings.allband_file
    
    if settings.dataset_mode == "regression_output":  # < for Four Corners
    
        ch4_file = settings.data_base_path + filename + settings.ch4_file
        mask_file = settings.data_base_path + filename + settings.mask_file

        with rasterio.open(ch4_file) as src:
            ch4_data = src.read(4, window=window)
        with rasterio.open(mask_file) as src:
            mask_data = src.read(4, window=window)
        mask_data = mask_data / 255.0 # to 0 or 1.0

        # TEMPORARILY set the labels as the CH4 products ...
        # label = mask_data * ch4_data
        label = ch4_data
        label_1ch = label.reshape((1,)+label.shape)

    if settings.dataset_mode == "segmentation_output":  # < for Permian Basin
        label_file = settings.data_base_path + filename + settings.label_file

        with rasterio.open(label_file) as src:
            # read(4) # <band 4 contains a concat of other, is very noisy
            label = src.read(1, window=window)
        label_1ch = label.reshape((1,)+label.shape)

        THR = 200
        label_1ch = np.where(label_1ch>THR, 1.0, 0.0)
        # label_1ch[label_1ch > 0.0] = 1.0 # convert into classification labels
        # label_1ch[label_1ch <= 0.0] = 0.0

    if load_x:
        with rasterio.open(allband_file) as src:
            # Select bands intelligently ...
            allbands_data = src.read([b+1 for b in bandlist], window=window)
            
            # # HAX - we loaded 3 bands - try the divisions
            # # 2310/2350 and 2350/2360
            # ba = allbands_data[0]
            # bb = allbands_data[1]
            # bd = allbands_data[2]
            # allbands_data = np.asarray([ba/bd, bb/bd, # < the divided ones
            #                               ba, bb, bd])  # < also the source ones
            # # btw: doesn't work well with the normalization

        tile_X = allbands_data
    else:
        tile_X = None

    label_1ch = label_1ch.astype(np.float32)
    tile_Y = label_1ch
    
    return tile_X, tile_Y

def get_filesystem(path):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0],requester_pays = True)
    else:
        # use local filesystem
        return fsspec.filesystem("file")
