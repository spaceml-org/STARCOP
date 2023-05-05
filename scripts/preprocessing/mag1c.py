from starcop.process_aviris import run_mag1c
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rdn', type=str, metavar='RADIANCE_FOLDER',
                        help='Folder with AVIRIS-NG image. It expects that the image and glt file have the same name as '
                             'the folder with suffix _img and _glt respectively')
    parser.add_argument('--out', type=str, metavar='OUTPUT_FILE',
                        help='File to write output into (geotiff) example: output.tif')
    parser.add_argument('--use_wavelength_range', type=float, default=(2122, 2488), nargs=2, metavar=('MIN', 'MAX'),
                        help='defines what contiguous range of wavelengths (in nanometers) should be included in the '
                             'filter calculation (default: %(default)s nm)')
    parser.add_argument('--samples_read', type=int, default=50,
                        help='Number of columns to read from AVIRIS (default: %(default)s)')

    args = parser.parse_args()

    start = time.time()
    run_mag1c(aviris_img_folder=args.rdn,
              mf_filename=args.out,
              use_wavelength_range=args.use_wavelength_range,
              samples_read=args.samples_read)

    time_elapsed = time.time()-start

    print(f"Image {args.rdn} processed in {time_elapsed:.0f} seconds")