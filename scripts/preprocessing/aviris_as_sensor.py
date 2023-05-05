from starcop.process_aviris import aviris_as_sensor
import argparse
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="AVIRIS to Sentinel-2 image conversion. "
                                           "This program uses the SRF of Sentinel-2 to simulate Sentinel-2 radiances "
                                           "from AVIRIS radiances")
    parser.add_argument('--rdn', type=str, required=True,
                        help='Folder with AVIRIS-NG image. It expects that the image and glt file have the same name as '
                             'the folder with suffix _img and _glt respectively')
    parser.add_argument('--out', type=str, required=True,
                        help='Folder to write S2 bands outputs')
    parser.add_argument('--sensor', type=str, default="S2", choices=["all", "S2", "S2A", "S2B", "WV3"],
                        help='Sensor to simulate ["all","S2", "S2A", "S2B", "WV3"]')

    args = parser.parse_args()

    if args.sensor == "all":
        sensors = ["WV3", "S2A", "S2B"]
    elif args.sensor == "S2":
        sensors = ["S2A", "S2B"]
    else:
        sensors = [args.sensor]

    print(f"Converting AVIRIS image {args.rdn} to sensors: {sensors}")
    start = time.time()
    aviris_as_sensor(aviris_img_folder_or_path=args.rdn, folder_dest=args.out, sensors=sensors, disable_pbar=False)
    time_elapsed = time.time() - start

    print(f"Image {args.rdn} processed in {time_elapsed:.0f} seconds")