#!/usr/bin/python3

# run this script as the device connects with dhcp.
# fill the folder /devices/mac with the device's data and start the collection for that device (inform the main process with tcp)
from socket import * 
import sys
import configparser
import os
import re
import time
import subprocess

hardcoded_devices = {'38:42:b:68:76:30': 'sonos_speaker',
 '90:f8:2e:87:c1:2e': 'amazon_echo_dot_5',
 '5c:47:5e:60:23:01': 'ring_doorbell',
 'a4:11:62:e5:2c:53': 'arlo_chime_01',
 'a4:97:5c:a4:6b:c4': 'vtech_baby_monitor',
 'a4:97:5c:a4:6b:c2': 'vtech_baby_camera',
 '04:17:b6:7f:b0:c2': 'eufy_chime',
 '04:17:b6:55:13:3a': 'eufy_camera',
 '70:3a:2d:38:1f:fa': 'geree_doorbell',
 '58:d3:49:49:77:31': 'apple_homepod',
 'fc:9c:98:11:71:a8': 'arlo_pro_4',
 '88:28:7d:43:80:c0': 'boifun_baby',
 'ac:bf:71:6b:e8:5b': 'bose_home_speaker',
 'dc:97:58:ab:a3:63': 'xiaomi_tv_box_s',
 '28:7e:80:96:8d:91': 'tcl_roku_tv',
 'b0:99:d7:2d:54:20': 'samsung_tv',
 'f4:03:2a:72:cb:f0': 'amazon_echo_flex',
 'b8:16:5f:ef:fb:58': 'lg_tv',
 '90:98:77:7b:6c:13': 'toshiba_tv',
 'cc:f7:35:49:f4:05': 'amazon_echo_dot_4',
 '90:48:6c:5e:e7:15': 'ring_chime_pro',
 '08:fb:ea:2e:2c:82': 'simplisafe_doorbell',
 'dc:e5:5b:5e:1b:f1': 'google_nest_hub',
 'd0:3f:27:82:15:5b': 'wyze_cam_pan_v2',
 '14:c9:cf:4a:6d:a9': 'blurams_a31',
 '44:42:01:4f:0f:8a': 'blink_mini',
 '38:86:f7:79:1d:15': 'google_nest_camera',
 '0c:8c:24:0b:be:fb': 'yi_camera',
 '08:12:a5:2d:27:bc': 'fire_tv_stick',
 '08:fb:ea:15:7d:ef': 'simplisafe_camera',
 '5c:47:5e:26:a4:11': 'ring_indoor_cam',
 '48:b0:2d:e9:9a:b9': 'nvidia_shield',
 'd8:eb:46:70:8d:4a': 'google_nest_doorbell',
 '70:ee:50:94:45:98': 'netatmo_doorbell',
 '08:e9:f6:2a:2e:a2': 'lavazza_coffee',
 'fc:67:1f:a9:a7:8d': 'weekett_kettle',
 '9c:9c:1f:91:e1:b6': 'cosori_airfrier',
 '0c:8b:95:a0:e2:cc': 'swan_kettle',
 'c4:dd:57:4a:4a:9c': 'levoit_purifier',
 '7c:c2:94:0b:da:4c': 'xiaomi_blender',
 'f4:cf:a2:4a:bd:46': 'blueair_air_purifier',
 '48:a2:e6:c9:80:f1': 'honeywell_thermostat',
 '3c:61:05:fd:32:ee': 'eufy_robovac_30c',
 '68:4e:05:9d:44:b8': 'ecovacs_deepbot_n8',
 'a0:92:08:86:e2:43': 'okp_vacuum',
 '48:e1:e9:ab:ec:1b': 'meross_garage_msg100',
 '00:24:e4:fa:ec:5c': 'withings_scale',
 '5c:d6:1f:e9:a9:c4': 'qardiobase_scale',
 '90:f1:57:a5:69:0f': 'garmin_scale',
 '48:55:19:0a:1b:4c': 'switchbot_hub_mini',
 '54:ef:44:3c:72:e8': 'aqara_hub',
 'a4:11:62:f4:08:30': 'arlo_base_station',
 'e4:5e:1b:de:58:f0': 'nest_wifi_router',
 '48:e1:e9:c3:d2:4f': 'meross_mss315',
 '28:6d:97:91:d7:ac': 'aeotec_hub',
 'bc:ff:4d:84:66:33': 'switchbot_hub_mini',
 '58:ef:68:99:7d:ed': 'belkin_plug',
 '48:e7:da:cd:de:a1': 'petsafe_feeder',
 '00:2d:b3:02:0e:70': 'furbo_360_camera',
 'd4:ad:fc:60:43:9a': 'govee_strip_light',
 '00:55:da:5f:9c:a8': 'nanoleaf_triangles',
 'ec:4d:3e:16:9b:e9': 'yeelight_yldp005',
 'ec:4d:3e:36:e1:bb': 'yeelight_yldp005',
 'ec:4d:3e:15:8f:bf': 'yeelight_yldp005',
 'd0:73:d5:5b:c8:23': 'lifx_mini',
 'd8:a0:11:49:ef:50': 'wiz_bulb',
 'ec:4d:3e:16:12:39': 'yeelight_yldp005',
 '4c:a9:19:ed:6b:c1': 'antela_bulb',
 'dc:62:79:d3:4f:35': 'tapo_l530e',
 '38:a5:c9:af:49:dd': 'fitop_bulb',
 '94:3a:91:db:fd:61': 'amazon_echo_dot_4',
 '68:9a:87:a5:fa:25': 'amazon_echo_dot_3',
 'b0:fc:0d:c3:30:8d': 'amazon_echo_dot_2',
 '68:37:e9:01:5a:23': 'amazon_echo_dot_2',
 '5c:41:5a:29:ad:97': 'amazon_echo_spot',
 '14:c1:4e:b0:93:7e': 'google_home_mini',
 '14:c1:4e:d5:8f:2c': 'google_home_mini',
 '54:60:09:6f:32:84': 'google_home_mini',
 '9c:8e:cd:17:26:6e': 'amcrest_prohd',
 '14:6b:9c:c6:ec:a6': 'wansview_camera',
 '0c:8c:24:a0:6f:47': 'genbolt_gb100s',
 '30:4a:26:9c:ed:30': 'lefun_camera',
 '70:ee:50:36:98:da': 'netatmo_weather',
 '50:c7:bf:b1:d2:78': 'tapo_p100',
 'b0:be:76:be:f2:aa': 'tapo_p100',
 'ec:75:0c:53:a8:f4': 'tapo_l900',
 'ec:75:0c:53:a8:f3': 'tapo_l900',
 '90:9a:4a:61:a2:6e': 'tapo_c100',
 '3c:28:6d:20:a0:a3': 'google_pixel3',
 '54:f1:5f:14:13:23': 'ezviz_c6n',
 '14:c1:4e:a2:f6:39': 'google_home_mini',
 'fc:2a:46:98:3b:b5': 'realme_12_pro_plus',
 '48:e1:e9:c3:d8:6a': 'meross_mss315',
 'bc:df:58:5f:bd:ca': 'google_chromecast',
 'c4:3c:b0:62:c6:41': 'reolink_doorbell',
 '48:e1:e9:3e:6e:17': 'meross_msl120',
 '38:42:0b:68:76:30': 'sonos_speaker',
 '00:24:e4:d6:69:e4': 'withings_thermo',
 '00:24:e4:ee:3e:bc': 'withings_bp',
 '00:24:e4:f1:ae:12': 'withings_sleep',
 '00:5f:bf:fa:50:02': 'omron_rs3',
 'df:57:3a:27:c3:9b': 'wellue_bp2a0256',
 '34:81:f4:f7:b2:83': 'ihealth_wrist',
 '5c:75:af:67:de:32': 'fitbit_versa_2',
 'a6:d3:ab:66:b7:98': 'galaxy_watch_5',
 'd2:43:7f:d2:57:ae': 'pixel_watch',
 '48:e1:e9:ed:f4:bf': 'meross_garage_msg100',
 '10:2c:b1:ef:e3:38':'eufy_c220',
  '2c:d8:de:76:7d:5b':'eufy_c220'
  }

# read the configuration file
config = configparser.ConfigParser()
config.read("/opt/mulini/config.ini")

# get the device folder
try:
    basic_folder = config['GENERAL']['BasicFolder']
except:
    sys.exit("Error: BasicFolder not found in config.ini")
    

db_file = basic_folder + "/" + config['DB']['DBFile']
log_file = basic_folder + "/logs/" + config['DHCP']['LogFile']

COOLDOWN_TIME = 60

def is_recently_processed(dev_mac,cooldown_time=COOLDOWN_TIME):
    # Check if the hardware address is already in the log and if it's within the cooldown period
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                try:
                    stored_hw_addr, timestamp = line.strip().split(',')
                    # Convert timestamp back to float
                    timestamp = float(timestamp)
                    # Check if the current time minus the timestamp is less than the cooldown time
                    if stored_hw_addr == dev_mac and (time.time() - timestamp) < COOLDOWN_TIME:
                        return True
                except:
                    continue
    return False

def log_processed(dev_mac):
    # Log the hardware address and current timestamp
    with open(log_file, 'a') as f:
        f.write(f"{dev_mac},{time.time()}\n")

def get_mac_vendor(dev_mac):
    import requests
    dev_mac_modified = ":".join(dev_mac.split(":")[:3]) + ":AA:AA:AA"
    response = requests.get(f"https://api.macvendors.com/{dev_mac_modified}")
    if response.status_code == 200:
        return response.text
    else:
        return "Unknown Vendor"
    

# Function to identify the device
def identify_device(dev_mac):
    default_type = "generic_smartphone"
    
    # perform device identification
    time.sleep(10)
    
    # collect traffic for some time and take all info...
    # then choose the type
    if dev_mac in hardcoded_devices:
        return hardcoded_devices[dev_mac]
    
    return default_type
    
# Function to check if the MAC address is valid
def check_mac(dev_mac):
    # Define a regex pattern for a valid MAC address
    mac_pattern = re.compile(r"^([0-9A-Fa-f]{2}:){5}([0-9A-Fa-f]{2})$")
    
    # Check if the MAC address matches the pattern
    if mac_pattern.match(dev_mac):
        return True
    return False

def get_interfaces():
    result = subprocess.run(['ifconfig', '-a'], capture_output=True, text=True)
    interfaces = []
    for line in result.stdout.splitlines():
        if line and not line[0].isspace():
            # The interface name is the first token before a space or colon
            iface = line.split()[0].split(':')[0]
            interfaces.append(iface)
    return interfaces

def get_device_interface(dev_mac):
    # Get the list of interfaces
    interfaces = get_interfaces()
    
    # Check each interface for the device MAC address
    for iface in interfaces:
        try:
            # Use the 'iw' command to check connected devices on the interface
            cmd = f"iw dev {iface} station dump | grep Station | awk '{{print $2}}'"
            output = subprocess.check_output(cmd, shell=True, text=True)
            connected_macs = output.splitlines()
            
            if dev_mac in connected_macs:
                return iface
        except subprocess.CalledProcessError:
            continue  # If the command fails, skip to the next interface
    
    return "unknown_interface"

# Function to check if the device is already in the database
def check_dev_in_db(cursor, dev_mac):
    # check if the device is already in the db
    # returns a tuple (a,b) where a is True if the device is in the db, b is True if the device is not an unknown device
    cursor.execute("SELECT dev_type FROM devices WHERE mac=?", (dev_mac,))
    device = cursor.fetchone()
    
    if device is None:
        return (False, False)
    
    return (True, device[0] != "unknown_device")

# Function to adjust the MAC address format
def fix_mac(dev_mac):
    # Ensure each byte in the MAC address has two characters
    dev_mac = dev_mac.lower()
    dev_mac = ":".join(byte.zfill(2) for byte in dev_mac.split(":"))
    return dev_mac


def main(dev_mac, dev_ip, dev_name,interface=None):
    print("Called script for device: ",dev_mac)    
    dev_mac = fix_mac(dev_mac)
    

    # check if the device is a valid mac address
    if not check_mac(dev_mac):
        print("Invalid MAC address format. Exiting...")
        return
    

    # immediately check if the device has been processed under the cooldown
    if is_recently_processed(dev_mac):
        print("Device has been processed recently, exiting...")
        # If the device has been recently processed, exit the script
        sys.exit(0)
    
    # log in the recently processed file
    print("Logging device...")
    log_processed(dev_mac)
    
    # assign device interface
    if interface is None:
        interface =get_device_interface(dev_mac)
        print("Device interface: ",interface)
    
    #check if the device is already in the db
    import sqlite3
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    dev_exists, prev_identified = check_dev_in_db(cursor,dev_mac)
    
    # if the device is not in the db, add it
    if not dev_exists:
        print("Device not in the db, adding it...")
        # get the mac vendor
        mac_vendor = get_mac_vendor(dev_mac)
        dev_type = "loading_device"
        # insert the device into the db
        cursor.execute("INSERT INTO devices (mac, name, interface, dev_type, ip, dhcp_name, last_dhcp_date, mac_vendor) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (dev_mac, dev_name, interface, dev_type, dev_ip, dev_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), mac_vendor))
        conn.commit()
        print(f"Device {dev_mac} added to the database.")
        
    # if dev is there, update ip, dhcp_name, last_dhcp_date
    else:
        cursor.execute("UPDATE devices SET interface=?, ip=?, dhcp_name=?, last_dhcp_date=? WHERE mac=?", (interface, dev_ip, dev_name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), dev_mac))
        conn.commit()
        print(f"Device {dev_mac} updated in the database.")
    
    
    if not prev_identified:    
        # perform device identification (this may take a while)
        dev_type= identify_device(dev_mac)
        
        #check if dev_type is in the db
        cursor.execute("SELECT dev_type_id FROM device_types WHERE dev_type_id=?", (dev_type,))
        dev_type_exists = cursor.fetchone()
        if not dev_type_exists:
            dev_type = "unknown_device"
        
        # Generate a unique device name based on the device type and count of devices with the same type
        cursor.execute("""
            SELECT dt.type_name, COUNT(d.mac) 
            FROM device_types dt 
            LEFT JOIN devices d ON dt.dev_type_id = d.dev_type 
            WHERE dt.dev_type_id = ?
        """, (dev_type,))
        dev_type_name, count = cursor.fetchone()
        dev_name = f"{dev_type_name} ({count + 1})"
        print("device name: ",dev_name)
        
        #update type in the db
        cursor.execute("""
            UPDATE devices 
            SET dev_type=?,
            name=?
            WHERE mac=?
        """, (dev_type, dev_name,dev_mac))

    conn.commit()
    # close the connection
    conn.close()
    
    
if __name__ == "__main__":
    # if the mac is not passed-> Error
    if len(sys.argv) < 4:
        print("Usage: python3 on-connect.py <device_mac> <device_ip> <device_name> [interface]")
        sys.exit(1)
    try:
        interface = sys.argv[4]
    except IndexError:
        interface = None
    # if the ip is not passed, use None
    main(sys.argv[1],sys.argv[2],sys.argv[3],interface=interface)
