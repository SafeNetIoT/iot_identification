import sys
import argparse
import requests
import time
import pandas as pd

VENDOR_API_URL = "https://api.macvendors.com/"
def get_manufacturer(address):
    try:
        response = requests.get(VENDOR_API_URL + address)
        if response.status_code != 200 or "error" in response.text.lower():
            return "Not Found"
        return response.text.strip()
    except Exception as e:
        return "Error"

def csv_lookup(input_file, output_file, column_name):
    res = []

    input_data = pd.read_csv(input_file)
    if column_name not in input_data.columns:
        print(f"Column '{column_name}' not found in input.")
        return

    for mac in input_data[column_name]:
        mac = str(mac).strip()
        manufacturer = get_manufacturer(mac)
        status = manufacturer != "Not Found"
        res.append([mac, manufacturer, status])
        print(status)
        time.sleep(1)

    res_df = pd.DataFrame(res, columns=['mac_address', 'manufacturer', 'status'])
    res_df.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="MAC address vendor lookup tool")
    parser.add_argument('mac', nargs='?', help="Single MAC address to look up")
    parser.add_argument('--input', help="CSV file with MAC addresses")
    parser.add_argument('--output', help="Output CSV file name")
    parser.add_argument('--column', help="Column name containing MAC addresses")


    args = parser.parse_args()

    if args.mac:
        print(get_manufacturer(args.mac))
    elif args.input:
        print('running csv tests...')
        csv_lookup(args.input, args.output, args.column)
    else:
        print("Error: Provide a MAC address or an input CSV file.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()        

        